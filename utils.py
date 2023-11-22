import glob
import pandas as pd
import random
import os
import torchvision.transforms as transforms
import torch
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing import Tuple
from dataset import BreastCancer


def create_dataset_csv(root_dir: str):
    """
    create the csv table with columns:
    - img_path: file path to each image
    - label: main type of image (benign/malignant)
    - subtype: subtypes of benign and malignant tumors (A/F/PT/TA/MC/DC/LC/PC)
    - magnification: maginification of each image (40X/100X/200X/400X)
    - split: split the dataset into train:eval:test = 60:20:20

    """

    img_paths = find_images(root_dir)
    df = pd.DataFrame(img_paths, columns=["img_path"])
    df["label"] = df["img_path"].str.split("/", expand=True).iloc[:, 8]
    df["subtype"] = (
        df["img_path"]
        .str.split("/", expand=True)
        .iloc[:, 10]
        .str.split("_", expand=True)
        .iloc[:, 2]
    )
    df["magnification"] = df["img_path"].str.split("/", expand=True).iloc[:, 11]
    #### create a dataframe consists of 4 cols: img_path, label, magnification, subtype and split (train/eval/test)    ## e.g.,
    ##   img_path                                        label           subtype     magnification    split
    ## '/.../SOB_M_MC-14-13418DE-100-009.png'           benign              MC             100X       train
    ## '/.../SOB_M_MC-14-13418DE-100-008.png'           benign              MC             100X       train
    ## '/.../SOB_M_MC-14-13418DE-100-003.png'           benign              MC             100X       train
    train_len = int(len(df) * 0.6)
    valid_len = int(len(df) * 0.2)
    split = (
        ["train"] * train_len + ["valid"] * valid_len + ["test"] * (len(df) - train_len - valid_len)
    )
    random.shuffle(split)
    df["split"] = split
    return df


def compute_accuracy(logits: Tensor, labels: Tensor, batch_size: int):  ## accuracy of 1 batch
    corrects = (torch.max(logits, 1)[1].view(labels.size()).data == labels.data).sum()
    accuracy = 100.0 * corrects / batch_size
    return accuracy.item()


def find_images(directory, image_extensions=["*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp"]):
    """
    Recursively finds all images in the directory with specified extensions.
    :param directory: The directory to search in.
    :param image_extensions: List of image file extensions to search for.
    :return: List of paths to the images found.
    """
    image_paths = []
    for extension in image_extensions:
        image_paths.extend(glob.glob(os.path.join(directory, "**", extension), recursive=True))
    return image_paths


def get_mean_std(dataloader):
    mean_img = 0
    mean_squared = 0
    num_batches = len(dataloader)
    for images, _ in tqdm(dataloader):
        mean_img += images.mean(dim=(0, 2, 3))
        mean_squared += images.mean(dim=(0, 2, 3)) ** 2
    mean = mean_img / num_batches
    # std = sqrt(E[X^2] - E[X]^2)
    std = mean_squared / num_batches - mean**2
    return mean, std


def get_dataloaders(batch_size: int, img_size: Tuple):
    ## prepare dataset
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Resize(img_size),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    trainset = BreastCancer(
        csv_path="breast_cancer_meta_data.csv",
        split="train",
        transform=transform,
    )
    validset = BreastCancer(
        csv_path="breast_cancer_meta_data.csv",
        split="valid",
        transform=transform,
    )
    testset = BreastCancer(
        csv_path="breast_cancer_meta_data.csv",
        split="test",
        transform=transform,
    )

    trainloader = DataLoader(
        trainset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=True
    )
    validloader = DataLoader(
        validset, batch_size=batch_size, shuffle=True, num_workers=2, drop_last=False
    )

    testloader = DataLoader(
        testset, batch_size=batch_size, shuffle=False, num_workers=2, drop_last=False
    )
    return trainloader, validloader, testloader


if __name__ == "__main__":  ### put all test code in this block
    folder_path = "/Users/vy/Documents/NEU_course/NN_course/breast_cancer_NN_project/data_model/"
    df = create_dataset_csv(folder_path)
    print(df.sample(10))
    df.to_csv("breast_cancer_meta_data.csv")
