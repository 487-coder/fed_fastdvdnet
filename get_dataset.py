from pathlib import Path
import imageio.v3 as iio
from torch.utils.data import Dataset


class Train_dataset(Dataset):
    def __init__(self,root_dir):
        self.root_dir = root_dir
        self.path = sorted(Path(root_dir).glob("*.mp4"))
    def __getitem__(self,idx):
        video_path = self.path[idx]
        frames = iio.imread(str(video_path), plugin="pyav")
        return frames
    def __len__(self):
        return len(self.path)

