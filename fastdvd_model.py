import torch
import torch.nn as nn
import torch.nn.functional as F


class CvBlock(nn.Module):
    def __init__(self, in_ch, out_ch):
        super(CvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class InputCvBlock(nn.Module):
    '''(Conv with num_in_frames groups => BN => ReLU) + (Conv => BN => ReLU)'''
    def __init__(self, num_in_frames, out_ch):
        super(InputCvBlock, self).__init__()
        self.interm_ch = 30
        self.convblock = nn.Sequential(
            nn.Conv2d(num_in_frames * (3 + 1), num_in_frames * self.interm_ch,
                      kernel_size=3, padding=1, groups=num_in_frames, bias=False),
            nn.BatchNorm2d(num_in_frames * self.interm_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_in_frames * self.interm_ch, out_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.convblock(x)


class DownBlock(nn.Module):
    '''Downscale + (Conv2d => BN => ReLU)*2'''
    def __init__(self, in_ch, out_ch):
        super(DownBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, stride=2, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            CvBlock(out_ch, out_ch)
        )

    def forward(self, x):
        return self.convblock(x)


class UpBlock(nn.Module):
    '''(Conv2d => BN => ReLU)*2 + Upscale'''
    def __init__(self, in_ch, out_ch):
        super(UpBlock, self).__init__()
        self.convblock = nn.Sequential(
            CvBlock(in_ch, in_ch),
            nn.Conv2d(in_ch, out_ch * 4, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2)
        )

    def forward(self, x):
        return self.convblock(x)


class OutputCvBlock(nn.Module):
    '''Conv2d => BN => ReLU => Conv2d'''
    def __init__(self, in_ch, out_ch):
        super(OutputCvBlock, self).__init__()
        self.convblock = nn.Sequential(
            nn.Conv2d(in_ch, in_ch, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(in_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False)
        )

    def forward(self, x):
        return self.convblock(x)


class DenBlock(nn.Module):
    """Definition of the denoising block of FastDVDnet."""
    def __init__(self, num_input_frames=3):
        super(DenBlock, self).__init__()
        self.chs_lyr0 = 32
        self.chs_lyr1 = 64
        self.chs_lyr2 = 128

        self.inc = InputCvBlock(num_in_frames=num_input_frames, out_ch=self.chs_lyr0)
        self.downc0 = DownBlock(in_ch=self.chs_lyr0, out_ch=self.chs_lyr1)
        self.downc1 = DownBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr2)
        self.upc2 = UpBlock(in_ch=self.chs_lyr2, out_ch=self.chs_lyr1)
        self.upc1 = UpBlock(in_ch=self.chs_lyr1, out_ch=self.chs_lyr0)
        self.outc = OutputCvBlock(in_ch=self.chs_lyr0, out_ch=3)

        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, in0, in1, in2, noise_map):
        x0 = self.inc(torch.cat((in0, noise_map, in1, noise_map, in2, noise_map), dim=1))
        x1 = self.downc0(x0)
        x2 = self.downc1(x1)
        x2 = self.upc2(x2)
        x1 = self.upc1(x1 + x2)
        x = self.outc(x0 + x1)
        x = in1 - x  # Residual connection
        return x


class FastDVDnet(nn.Module):
    """Definition of the FastDVDnet model."""
    def __init__(self, num_input_frames=5):
        super(FastDVDnet, self).__init__()
        self.num_input_frames = num_input_frames
        self.temp1 = DenBlock(num_input_frames=3)
        self.temp2 = DenBlock(num_input_frames=3)
        self.reset_params()

    @staticmethod
    def weight_init(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight, nonlinearity='relu')

    def reset_params(self):
        for m in self.modules():
            self.weight_init(m)

    def forward(self, x, noise_map):
        x0, x1, x2, x3, x4 = [x[:, 3 * i:3 * i + 3, :, :] for i in range(self.num_input_frames)]
        x20 = self.temp1(x0, x1, x2, noise_map)
        x21 = self.temp1(x1, x2, x3, noise_map)
        x22 = self.temp1(x2, x3, x4, noise_map)
        x = self.temp2(x20, x21, x22, noise_map)
        return x


def frame_denoise(model, noise_frame, sigma_map, context):
    _, _, h, w = noise_frame.shape
    pad_h = (4 - h % 4) % 4
    pad_w = (4 - w % 4) % 4
    if pad_h or pad_w:
        noise_frame = F.pad(noise_frame, (0, pad_w, 0, pad_h), mode="reflect")
        sigma_map = F.pad(sigma_map, (0, pad_w, 0, pad_h), mode="reflect")
    with context:
        denoise_frame = model(noise_frame, sigma_map)
        denoise_frame = torch.clamp(denoise_frame, 0.0, 1.0)
    if pad_h:
        denoise_frame = denoise_frame[:, :, :-pad_h, :]
    if pad_w:
        denoise_frame = denoise_frame[:, :, :, :-pad_w]
    return denoise_frame


def denoise_seq_fastdvdnet(seq, noise_std, model, temporal_window=5, is_training=False):
    frame_num, c, h, w = seq.shape
    center = (temporal_window - 1) // 2
    denoise_frames = torch.empty_like(seq).to(seq.device)
    noise_map = noise_std.view(1, 1, 1, 1).expand(1, 1, h, w).to(seq.device)
    model.to(seq.device)
    context = torch.enable_grad() if is_training else torch.no_grad()
    frames = []

    with context:
        for denoise_index in range(frame_num):
            if not frames:
                for index in range(temporal_window):
                    rel_index = abs(index - center)
                    frames.append(seq[rel_index])
            else:
                del frames[0]
                rel_index = min(denoise_index + center,
                                -denoise_index + 2 * (frame_num - 1) - center)
                frames.append(seq[rel_index])

            input_tensor = torch.stack(frames, dim=0).view(1, temporal_window * c, h, w).to(seq.device)
            denoise_frames[denoise_index] = frame_denoise(model, input_tensor, noise_map, context)

        torch.cuda.empty_cache()
        return denoise_frames
