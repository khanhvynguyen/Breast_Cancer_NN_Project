import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import os

from utils import get_dataloaders, print_out
from models.CNN import MyCNN
from train import train_one_epoch
from eval import eval_model
from models.resnet import make_resnet18
from models.MLP import MLPModel


def main(model_type: str):
    """
    model_type: MLP or CNN or ResNet
    """
    ## Hyperparameters
    n_epochs = 10
    batch_size = 32
    classes = ("benign", "malignant")
    n_channels = 3  ### red, green, blue
    h, w = 224, 224
    img_size = (h, w)
    n_classes = len(classes)
    lr = 0.001
    momentum = 0.9
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    kernel_size = 3
    input_dim = img_size[0]  ## h=w=input_dim
    flatten = False
    k1 = 32  # number of kernels in conv_1
    k2 = 32  # number of kernels in conv_2
    k3 = 32  # number of kernels in conv_3

    ## create a new folder to store log files named logs
    os.makedirs("logs", exist_ok=True)

    datetime_now = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/resnet18_epoch{n_epochs}_lr{lr}_{datetime_now}.log"
    log = open(log_filename, "a")

    ## prepare datasets
    trainloader, validloader, testloader = get_dataloaders(batch_size, img_size)

    if model_type == "MLP":
        net = MLPModel(h, w)
    elif model_type == "CNN":
        net = MyCNN(
            input_dim=input_dim,
            num_kernel_conv1=k1,
            num_kernel_conv2=k2,
            num_kernel_conv3=k3,
            n_classes=n_classes,
            kernel_size=kernel_size,
        )
    elif model_type == "ResNet":
        net = make_resnet18(input_dim=n_channels, num_classes=2)
    else:
        raise NotImplementedError()

    optimizer = torch.optim.Adam(  ## optimizer: the way to update thetas, we are using SGD way. Besides, there is also Adam,...
        net.parameters(), lr=lr
    )
    loss = torch.nn.CrossEntropyLoss()  ### Compute Loss (CELoss, MSE)

    train_acc_summary = []
    train_loss_summary = []
    valid_acc_summary = []
    valid_loss_summary = []

    for e in range(n_epochs):
        start_time = time.time()  ## time at the beginning of epoch, for logging purpose
        train_acc, train_loss = train_one_epoch(
            model=net,
            device=device,
            trainloader=trainloader,
            optimizer=optimizer,
            criterion=loss,
            flatten=flatten,
        )

        eval_acc, eval_loss = eval_model(
            model=net,
            device=device,
            evalloader=validloader,
            criterion=loss,
            flatten=flatten,
        )

        train_acc_summary.append(train_acc)
        train_loss_summary.append(train_loss)
        valid_acc_summary.append(eval_acc)
        valid_loss_summary.append(eval_loss)

        end_time = time.time()  ## time at the end of epoch
        runtime = end_time - start_time  ## runtime of 1 epoch, in seconds
        runtime_mins = round(runtime / 60, 1)  ## runtime of 1 epoch, in minutes

        print_out(
            f"Epoch: {e+1} | Train loss: {train_loss} | Train acc: {train_acc} | Valid loss: {eval_loss} | Valid acc: {eval_acc}| Runtime: {runtime_mins} mins",
            log,
        )

    ## TODO: move the following code to a separate function in utils.py
    # Create table result
    df_epoch = pd.DataFrame({"epoch": range(1, n_epochs + 1)})
    df_train = pd.DataFrame(train_acc_summary)
    df_eval = pd.DataFrame(valid_acc_summary)
    df_train_summary = pd.concat([df_epoch, df_train], axis=1)
    df_eval_summary = pd.concat([df_epoch, df_eval], axis=1)

    print_out(f"Train summary: {df_train_summary}", log)
    print_out(f"Eval summary: {df_eval_summary}", log)

    # Visulaize the results
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


if __name__ == "__main__":
    model_types = ["MLP", "CNN", "ResNet"]
    for model_type in model_types:
        print(f"Training {model_type}...")
        main(model_type=model_type)
        print("----------------------------------------------")
