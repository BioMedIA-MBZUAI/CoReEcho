import torch

from .augmentation import get_augmentation
from .dataset import EchoNet, EchoNetRNC

def set_loader(opt, quiet=False):
    train_transform = None
    if opt.aug == True:
        train_transform = get_augmentation(opt.frames)
    
    if not quiet:
        print(f'Train transform: {train_transform}')
    
    train_ds = EchoNetRNC(
            root=opt.data_folder,
            split="train",
            frames=opt.frames,
            frequency=opt.frequency,
            transform=train_transform,
    )
    
    train_no_aug_ds = EchoNet(
            root=opt.data_folder,
            split="train",
            frames=opt.frames,
            frequency=opt.frequency,
    )
    val_ds = EchoNet(
            root=opt.data_folder,
            split="val",
            frames=opt.frames,
            frequency=opt.frequency,
    )
    test_ds = EchoNet(
            root=opt.data_folder,
            split="test",
            frames=opt.frames,
            frequency=opt.frequency,
    )
    
    if not quiet:
        print(f'Train set size: {train_no_aug_ds.__len__()}')
        print(f'Valid set size: {val_ds.__len__()}')
        print(f'Test set size: {test_ds.__len__()}')
    
    train_loader = torch.utils.data.DataLoader(
        train_ds, batch_size=opt.batch_size, shuffle=True, num_workers=opt.num_workers, pin_memory=True, drop_last=True,
    )
    train_no_aug_loader = torch.utils.data.DataLoader(
        train_no_aug_ds, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_ds, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers, pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_ds, batch_size=opt.batch_size, shuffle=False, num_workers=opt.num_workers,
    )
    
    return train_loader, train_no_aug_loader, val_loader, test_loader