import tonic
import tonic.transforms as transforms
import torch
from torch.utils.data import DataLoader

def get_shd_dataloaders(batch_size=64, num_workers=0):
    # 把事件流转换成固定时间步的spike tensor
    # 100个时间步，700个输入通道（SHD有700个听觉神经元）
    transform = transforms.ToFrame(
        sensor_size=tonic.datasets.SHD.sensor_size,
        n_time_bins=100
    )

    train_dataset = tonic.datasets.SHD(
        save_to="./data/raw",
        train=True,
        transform=transform
    )

    test_dataset = tonic.datasets.SHD(
        save_to="./data/raw",
        train=False,
        transform=transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=tonic.collation.PadTensors(batch_first=True)
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=tonic.collation.PadTensors(batch_first=True)
    )

    return train_loader, test_loader