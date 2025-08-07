import argparse
import time
from fastdvd_model import FastDVDnet
from fastdvd_model import denoise_seq_fastdvdnet
from utils import *

NUM_IN_FR_EXT = 5  # temporal size of patch
OUTIMGEXT = '.png'  # output images format




def test_fastdvdnet(**kwargs):
    # Start timer
    start_time = time.time()

    # Setup path
    save_dir = Path(kwargs['save_path'])
    save_dir.mkdir(parents=True, exist_ok=True)

    # Setup logger
    logger = init_logger_test(kwargs['save_path'])

    # Select device
    device = torch.device('cuda' if kwargs['cuda'] and torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")

    # Load model
    logger.info("Loading model...")
    model = FastDVDnet(num_input_frames=NUM_IN_FR_EXT)

    state_dict = torch.load(kwargs['model_file'], map_location=device, weights_only=True)
    if device.type == 'cuda':
        model = nn.DataParallel(model).to(device)
    #else:
        #state_dict = remove_parallel_wrapper(state_dict)
    model.load_state_dict(state_dict)
    model.eval()

    # Load sequence
    seq, _, _ = open_sequence(kwargs['test_path'], kwargs['gray'], expand_if_needed=False,
                              max_num_fr=kwargs['max_num_fr_per_seq'])
    seq = torch.from_numpy(seq).to(device)
    if seq.dim() == 5 and seq.shape[0] == 1:
        seq = seq.squeeze(0)
    print(seq.min(), seq.max())

    # Add noise
    noise = torch.empty_like(seq).normal_(mean=0, std=kwargs['noise_sigma']).to(device)
    noisy_seq = seq + noise
    noise_map = torch.tensor([kwargs['noise_sigma']], dtype=torch.float32).to(device)

    logger.info(f"Sequence loaded. Shape: {seq.shape}, Noise sigma: {kwargs['noise_sigma']}")

    denoised_seq = denoise_seq_fastdvdnet(
        seq=noisy_seq,
        noise_std=noise_map,
        temporal_window=NUM_IN_FR_EXT,
        model=model
    )
    '''
    # Save output images if required
    if not kwargs['dont_save_results']:
        save_out_seq(
            seqnoisy=noisy_seq,
            seqclean=denoised_seq,
            save_dir=save_dir,
            sigmaval=int(kwargs['noise_sigma'] * 255),
            suffix=kwargs['suffix'],
            save_noisy=True
        )'''

    total_time = time.time() - start_time

    psnr = batch_psnr(denoised_seq, seq, 1.)
    psnr_noisy = batch_psnr(noisy_seq.squeeze(), seq, 1.)
    logger.info("Finished denoising {}".format(kwargs['test_path']))
    logger.info(f"PSNR noisy {psnr_noisy:.4f}dB, PSNR result {psnr:.4f}dB\n")
    logger.info(f"Finished denoising in {total_time:.2f} seconds.")


if __name__ == "__main__":
    # Parse arguments
    parser = argparse.ArgumentParser(description="Denoise a sequence with FastDVDnet")
    parser.add_argument("--model_file", type=str,
                        default="./model.pth",
                        help='path to model of the pretrained denoiser')
    parser.add_argument("--test_path", type=str, default="test_video/sunflower",
                        help='path to sequence to denoise')
    #parser.add_argument("--suffix", type=str, default="", help='suffix to add to output name')
    parser.add_argument("--max_num_fr_per_seq", type=int, default=25,
                        help='max number of frames to load per sequence')
    parser.add_argument("--noise_sigma", type=float, default=25, help='noise level used on test set')
    #parser.add_argument("--dont_save_results", action='store_true', help="don't save output images")
    #parser.add_argument("--save_noisy", action='store_true', help="save noisy frames")
    parser.add_argument("--gpu", default=True, help="run model on CPU")
    parser.add_argument("--save_path", type=str, default='./results',
                        help='where to save outputs as png')
    parser.add_argument("--gray", action='store_true',
                        help='perform denoising of grayscale images instead of RGB')

    args = parser.parse_args()
    args.noise_sigma /= 255.0

    args.cuda = args.gpu and torch.cuda.is_available()
    args_dict = vars(args)
    print("\n### Testing FastDVDnet model ###")
    print("> Parameters:")
    for k, v in args_dict.items():
        print(f'{k}: {v}')
    print('\n')

    test_fastdvdnet(**vars(args))
