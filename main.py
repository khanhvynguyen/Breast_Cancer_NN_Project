import torch
import time
import pandas as pd
import matplotlib.pyplot as plt
import os
import yaml
import argparse

from utils import get_dataloaders, print_out
from models.CNN import MyCNN
from train import train_one_epoch
from eval import eval_model
from models.resnet import make_resnet18
from models.MLP import MLPModel


def main(args):
    """
    model_type: MLP or CNN or ResNet
    """
    config_file_name = args.config_file_name

    ## read the config file
    with open(f"configs/{config_file_name}.yaml") as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    ## Hyperparameters
    n_epochs = config["n_epochs"]
    batch_size = config["batch_size"]
    h, w = config["h"], config["w"]
    lr = config["lr"]
    momentum = config["momentum"]
    device = config["device"]
    kernel_size = config["kernel_size"]
    flatten = config["flatten"]
    model = config["model"]
    kernel_list = config.get("kernel_list", None)
    optimizer = config["optimizer"]

    ## check if cuda is available
    if not torch.cuda.is_available():
        device = "cpu"

    img_size = (h, w)
    input_dim = h  ## h=w=input_dim
    classes = ["benign", "malignant"]
    n_classes = len(classes)
    n_channels = 3  ### red, green, blue

    ## create a new folder to store log files named logs
    os.makedirs("logs", exist_ok=True)

    datetime_now = time.strftime("%Y-%m-%d_%H-%M-%S")
    log_filename = f"logs/{model}_epoch{n_epochs}_lr{lr}_{datetime_now}.log"
    log = open(log_filename, "a")

    print_out(f"config: {config}", log)

    ## prepare datasets
    trainloader, validloader, testloader = get_dataloaders(batch_size, img_size)
    ## TODO: test on testloader at the end of training

    if model == "MLP":
        net = MLPModel(h, w)
    elif model == "CNN":
        net = MyCNN(
            input_dim=input_dim,
            num_kernel_conv1=kernel_list[0],
            num_kernel_conv2=kernel_list[1],
            num_kernel_conv3=kernel_list[2],
            n_classes=n_classes,
            kernel_size=kernel_size,
        )  # type: ignore
    elif model == "ResNet":
        net = make_resnet18(input_dim=n_channels, num_classes=2)
    else:
        raise NotImplementedError()

    # Optimizer
    if config["optimizer"] == "adam":
        optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    elif config["optimizer"] == "sgd":
        optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=momentum)
    else:
        raise NotImplementedError(f"Optimizer {config['optimizer']} not implemented")

    loss = torch.nn.CrossEntropyLoss()  ### Compute Loss (CELoss, MSE)

    train_acc_summary = []
    train_loss_summary = []
    valid_acc_summary = []
    valid_loss_summary = []

    best_valid_acc = 0.0
    final_test_acc = 0.0

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

        valid_acc, valid_loss = eval_model(
            model=net,
            device=device,
            evalloader=validloader,
            criterion=loss,
            flatten=flatten,
        )
        test_acc, test_loss = eval_model(
            model=net,
            device=device,
            evalloader=testloader,
            criterion=loss,
            flatten=flatten,
        )

        if valid_acc["avg_acc"] > best_valid_acc:
            best_valid_acc = valid_acc["avg_acc"]
            final_test_acc = test_acc

        train_acc_summary.append(train_acc)
        train_loss_summary.append(train_loss)
        valid_acc_summary.append(valid_acc)
        valid_loss_summary.append(valid_loss)

        end_time = time.time()  ## time at the end of epoch
        runtime = end_time - start_time  ## runtime of 1 epoch, in seconds
        runtime_mins = round(runtime / 60, 1)  ## runtime of 1 epoch, in minutes

        print_out(
            f"Epoch: {e+1} | Train loss: {train_loss} | Train acc: {train_acc} | Valid loss: {valid_loss} | Valid acc: {valid_acc}| Runtime: {runtime_mins} mins",
            log,
        )

    print_out(f"Final test accuracy: {final_test_acc}", log)

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
    ## create a parser object
    parser = argparse.ArgumentParser()

    parser.add_argument("--config_file_name", "-c", type=str, default="resnetv1")
    args = parser.parse_args()
    main(args)
