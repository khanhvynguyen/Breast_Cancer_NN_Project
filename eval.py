import torch
import torch.nn as nn
from torch.utils.data.dataloader import DataLoader
from utils import compute_accuracy
import numpy as np


@torch.no_grad()  # decorator
def eval_model(
    model: nn.Module,
    device: torch.device,
    evalloader: DataLoader,
    criterion: torch.nn.CrossEntropyLoss,
    flatten: bool,
    is_auc: bool = False,
):
    ## Set model to "evaluate" model
    model.eval()

    ## Keep track of loss and accuracy
    eval_loss = 0.0
    mapper = {"0": "40X", "1": "100X", "2": "200X", "3": "400X"}
    eval_acc = {"40X": 0.0, "100X": 0.0, "200X": 0.0, "400X": 0.0}

    ## Number of batches
    n_batches = len(evalloader)

    for i, (images, labels, magnifications) in enumerate(evalloader):
        # if images.shape != (4, 3, 20, 20):
        #     breakpoint()

        if flatten:
            images = images.reshape(images.shape[0], -1)
            # images = torch.flatten(start_dim=1)

        ## Move images and labels to `device` (CPU or GPU)
        images = images.to(device)
        labels = labels.to(device)
        magnifications = magnifications.to(device)
        ##### [YOUR CODE] Step 1. Forward pass: pass the data forward, the model try its best to predict what the output should be
        # You need to get the output from the model, store in a new variable named `logits`
        logits = model(images)  ## call forward funtion from class FirstNeuralNet

        ##### [YOUR CODE] Step 2. Compare the output that the model gives us with the real labels
        ## You need to compute the loss, store in a new variable named `loss`
        loss = criterion(logits, labels)

        # End of your code --------------------------------------------------------------------------------------------------------
        ## Compute loss and accuracy for this batch
        eval_loss += loss.detach().item()

        #  compute eval_acc based on magnification
        for i in range(4):
            logits_i = logits[magnifications == i]
            labels_i = labels[magnifications == i]
            batch_size_i = len(logits_i)
            if batch_size_i == 0:
                continue
            magnif = mapper[str(i)]
            eval_acc[magnif] += compute_accuracy(logits_i, labels_i, batch_size_i)

    eval_acc = {k: v / n_batches for k, v in eval_acc.items()}
    eval_loss = eval_loss / n_batches
    # compute average aval_acc
    eval_acc["avg_acc"] = np.mean(list(eval_acc.values()))

    ## TODO:
    if is_auc:
        ## plot AUC here
        pass
    return eval_acc, eval_loss
