import glob
import pandas as pd
import random
import os
import torchvision.transforms as transforms
import torch
import time
import matplotlib.pyplot as plt
import pprint
import yaml
import itertools
from typing import List, Dict
from torch import Tensor
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from typing import Tuple
from dataset import BreastCancer

from sklearn.model_selection import train_test_split


def create_dataset_csv(root_dir: str, seed: int = 2023):
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

    ## split the dataset into train:eval:test = 60:20:20, stratified by magnification
    train, valid_test = train_test_split(
        df, test_size=0.4, stratify=df["magnification"], random_state=seed
    )
    valid, test = train_test_split(
        valid_test, test_size=0.5, stratify=valid_test["magnification"], random_state=seed
    )
    train["split"] = "train"
    valid["split"] = "valid"
    test["split"] = "test"
    df = pd.concat([train, valid, test])

    return df


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
    for images, _,_ in tqdm(dataloader):
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
            transforms.Resize(img_size, antialias=True),
            transforms.Normalize((0.5719, 0.2510, 0.5272), (0.0005, 0.0011, 0.0004)),
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


def print_out(print_str, log):
    print(print_str)
    datetime_now = time.strftime("%Y-%m-%d_%H-%M-%S")
    ## append different models to log file:
    log.write(datetime_now + ": " + print_str + "\n")
    log.flush()


def plot_train_eval_summary(df_train_summary, df_eval_summary):
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))

    ax1.plot(df_train_summary["epoch"], df_train_summary["40X"], label="40X")
    ax1.plot(df_train_summary["epoch"], df_train_summary["100X"], label="100X")
    ax1.plot(df_train_summary["epoch"], df_train_summary["200X"], label="200X")
    ax1.plot(df_train_summary["epoch"], df_train_summary["400X"], label="400X")
    ax1.plot(
        df_train_summary["epoch"],
        df_train_summary["avg_acc"],
        label="Average Accuracy",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    ax1.set_title("Training accuracy over Epochs")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Accuracy (%)")
    ax1.legend()

    ax2.plot(df_eval_summary["epoch"], df_eval_summary["40X"], label="40X")
    ax2.plot(df_eval_summary["epoch"], df_eval_summary["100X"], label="100X")
    ax2.plot(df_eval_summary["epoch"], df_eval_summary["200X"], label="200X")
    ax2.plot(df_eval_summary["epoch"], df_eval_summary["400X"], label="400X")
    ax2.plot(
        df_eval_summary["epoch"],
        df_eval_summary["avg_acc"],
        label="Average Accuracy",
        color="black",
        linewidth=2,
        linestyle="--",
    )
    ax2.set_title("Training accuracy over Epochs")
    ax2.set_title("Evaluation accuracy over Epochs")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Accuracy (%)")
    ax2.legend()

    plt.show()



def get_grid_search(config_path: str, 
                    optimizers: list, 
                    learning_rates: list,
                    num_blocks_list:list,
                    is_batchnorm:list) -> List[Dict]:
    """
    Read the config file and return a list of all possible combinations of hyperparameters
    """
    ## read the config file
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    combinations = list(
        itertools.product(optimizers, learning_rates, num_blocks_list, is_batchnorm)
    )
    grid_search = []
    for opt, lr, num_blocks, bn in combinations:
        new_config = config.copy()
        new_config["optimizer"] = opt
        new_config["lr"] = lr
        new_config["num_blocks_list"] = num_blocks
        new_config["is_batchnorm"] = bn
        grid_search.append(new_config)

    return grid_search

if __name__ == "__main__":  ### put all test code in this block
    # folder_path = "data_model/"
    # df = create_dataset_csv(folder_path)
    # print(df.sample(30))
    # df.to_csv("breast_cancer_meta_data.csv")
    # Calculate mean and std of trainloader
    trainloader = get_dataloaders(batch_size=32, img_size=(224, 224))[0]
    mean, std = get_mean_std(trainloader)
    print(mean)
    print(std)
# tensor([0.5719, 0.2510, 0.5272])
# tensor([0.0005, 0.0011, 0.0004])