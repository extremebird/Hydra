
import torch.utils.data as data
from PIL import Image
import os
import os.path
from torchvision import transforms
import torch
from timm.data import create_transform

def default_loader(path):
    return Image.open(path).convert('RGB')


def default_flist_reader(flist):
    """
    flist format: impath label\nimpath label\n ...(same to caffe's filelist)
    """
    imlist = []
    with open(flist, 'r') as rf:
        for line in rf.readlines():
            impath, imlabel = line.strip().split()
            imlist.append((impath, int(imlabel)))

    return imlist


class ImageFilelist(data.Dataset):
    def __init__(self, root, flist, transform=None, target_transform=None,
                 flist_reader=default_flist_reader, loader=default_loader):
        self.root = root
        self.imlist = flist_reader(flist)
        self.transform = transform
        self.target_transform = target_transform
        self.loader = loader

    def __getitem__(self, index):
        impath, target = self.imlist[index]
        img = self.loader(os.path.join(self.root, impath))
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self):
        return len(self.imlist)


def get_data(name, evaluate=True, batch_size=64,seed=0,mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]):
    root='./data/' + name
    root_train = root
    root_val = root
    trainval_flist=root + "/train800val200.txt"
    train_flist=root +"/train800.txt"
    val_flist=root + "/val200.txt"
    test_flist=root + "/test.txt"
    train_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    val_transform = transforms.Compose([
            transforms.Resize((224, 224), interpolation=3),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)])
    if evaluate:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root_train, flist=trainval_flist,
                transform=train_transform),
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root_val, flist=test_flist,
                transform=val_transform),
            batch_size=256, shuffle=False,
            num_workers=4, pin_memory=True)
    else:
        train_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root_train, flist=train_flist,
                transform=train_transform),
            batch_size=batch_size, shuffle=True, drop_last=True,
            num_workers=4, pin_memory=True)

        val_loader = torch.utils.data.DataLoader(
            ImageFilelist(root=root_val, flist=val_flist,
                transform=val_transform),
            batch_size=256, shuffle=False,
            num_workers=4, pin_memory=True)
    return train_loader, val_loader

