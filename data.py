import torch
import torchvision
import torchvision.transforms.v2 as transforms

def get_split_dls(batch_size=128, num_workers=2, download=True, heavy_regularization=False):
    # values as per https://github.com/Armour/pytorch-nn-practice/blob/master/utils/meanstd.py
    
    # prepare list of augmentations : list[Callable]
    # probably using transforms.Lambda(lambda tau: aug_fn(x))
    if not heavy_regularization:
        transform_train = transforms.Compose(
            [
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True), 
                transforms.Normalize(
                    (0.49139968, 0.48215841, 0.44653091),
                    (0.24703223, 0.24348513, 0.26158784),
                ),
                # augmentations
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
                # transforms.GaussianBlur(kernel_size=5, sigma=0.25),
            ]
        )
    else:
        transform_train = transforms.Compose(
            [
                transforms.ToImage(), 
                transforms.ToDtype(torch.float32, scale=True), 
                # transforms.GaussianBlur(kernel_size=5, sigma=0.25),
                transforms.RandomCrop(32, padding=4),
                transforms.RandomVerticalFlip(0.5),
                transforms.RandomHorizontalFlip(0.5),
                transforms.Normalize(
                    (0.49139968, 0.48215841, 0.44653091),
                    (0.24703223, 0.24348513, 0.26158784),
                ),
            ]
        )        

    transform_test = transforms.Compose(
        [
            transforms.ToImage(), 
            transforms.ToDtype(torch.float32, scale=True),
            transforms.Normalize(
                (0.49139968, 0.48215841, 0.44653091),
                (0.24703223, 0.24348513, 0.26158784),
            ),
        ]
    )

    trainset = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=download, transform=transform_train
    )
    augmentations = []
    for aug in augmentations:
        aug_set = torchvision.datasets.CIFAR10(
        root="./data", train=True, download=download, transform=aug
    )
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    testset = torchvision.datasets.CIFAR10(
        root="./data", train=False, download=download, transform=transform_test
    )
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=100, shuffle=False, num_workers=num_workers
    )

    classes = (
        "plane",
        "car",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    )

    return trainloader, testloader, classes
