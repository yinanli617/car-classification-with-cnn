import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import torch.distributed as dist
import timm

import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm
import os
import PIL.Image as Image
import logging
import argparse

WORLD_SIZE = int(os.environ.get("WORLD_SIZE", 1))
MODEL_NAME = 'efficientnet_b3'
DATASET_DIR = './stanford_car_dataset/car_data/car_data/'
NUM_CAR_CLASSES = 196


def get_dataloaders(dataset_dir, batch_size, num_workers):
    # To help prevent overfitting, I did some simple augmentation including horizontal flip and rotation here.
    # For more image augmentation options, the albumentations package works really well with pytorch
    # https://github.com/albumentations-team/albumentations

    # On another note, I recently found another helpful package - torchIO which is good for 3d volume dataset
    # https://github.com/fepegar/torchio

    # note: no data augmentation for test data

    width, height = 224, 224
    train_tfms = transforms.Compose([transforms.Resize((width, height)),
                                     transforms.RandomHorizontalFlip(),
                                     transforms.RandomRotation(15),
                                     transforms.ToTensor(),
                                     transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    test_tfms = transforms.Compose([transforms.Resize((width, height)),
                                    transforms.ToTensor(),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # create datasets
    train_dset = torchvision.datasets.ImageFolder(root=dataset_dir + "train", transform=train_tfms)
    train_dl = torch.utils.data.DataLoader(train_dset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

    test_dset = torchvision.datasets.ImageFolder(root=dataset_dir + "test", transform=test_tfms)
    test_dl = torch.utils.data.DataLoader(test_dset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_dl, test_dl


def train_model(model, device, train_dl, test_dl, loss_fn, optimizer, scheduler, save_model, n_epochs=5):
    losses = []
    accuracies = []
    test_losses = []
    test_accuracies = []

    best_test_acc = 0
    # set the model to train mode initially
    model.train()
    for epoch in tqdm.tqdm(range(1, n_epochs + 1)):
        running_loss = 0.0
        running_correct = 0.0
        for i, data in enumerate(train_dl, 0):
            # get the inputs and assign them to cuda
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = model(inputs)
            predicted = F.softmax(outputs, dim=-1).argmax(dim=-1)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

            # calculate the loss/acc later
            running_loss += loss.item() * inputs.shape[0]
            running_correct += (labels == predicted).sum().item()

        epoch_loss = running_loss / len(train_dl.dataset)
        epoch_acc = running_correct / len(train_dl.dataset) * 100.0

        with logging_redirect_tqdm():
            msg = f"Train Epoch: {epoch}\ttrain_loss: {epoch_loss:.4f}\ttrain_accuracy: {epoch_acc:.4f}%"
            logging.info(msg)

        losses.append(epoch_loss)
        accuracies.append(epoch_acc)

        # switch the model to eval mode to evaluate on test data
        model.eval()
        test_loss, test_acc = eval_model(model, device, test_dl, loss_fn)
        test_losses.append(test_loss)
        test_accuracies.append(test_acc)

        if test_acc > best_test_acc and save_model:
            best_test_acc = test_acc
            torch.save(model.state_dict(), 'saved_model.pt')

        # re-set the model to train mode after validating
        model.train()
        scheduler.step(test_acc)
    print('Finished Training')
    return model, losses, accuracies, test_losses, test_accuracies


def eval_model(model, device, test_dl, loss_fn):
    correct = 0.0
    total = 0.0
    loss_total = 0.0
    with torch.no_grad():
        for i, data in enumerate(test_dl, 0):
            images, labels = data
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = loss_fn(outputs, labels)
            loss_total += loss.item() * images.shape[0]
            predicted = F.softmax(outputs, dim=-1).argmax(dim=-1)

            correct += (predicted == labels).sum().item()

    test_acc = 100.0 * correct / len(test_dl.dataset)
    test_loss = loss_total / len(test_dl.dataset)

    with logging_redirect_tqdm():
        msg = f'test_loss: {test_loss:.4f}\ttest_accuracy: {test_acc:.4f}%\n'
        logging.info(msg)

    return test_loss, test_acc


def main():
    # Training settings
    parser = argparse.ArgumentParser(description="Car classification with CNN and transfer learning")
    parser.add_argument("--epochs", type=int, default=20, metavar="N",
                        help="number of epochs to train (default: 20)")
    parser.add_argument("--batch-size", type=int, default=32, metavar="N",
                        help="batch size (default: 32)")
    parser.add_argument("--num-workers", type=int, default=2, metavar="N",
                        help="number of workers (default: 2)")
    parser.add_argument("--lr", type=float, default=0.001, metavar="LR",
                        help="learning rate (default: 0.001)")
    parser.add_argument("--no-cuda", action="store_true", default=False,
                        help="disables CUDA training")
    parser.add_argument("--seed", type=int, default=42, metavar="S",
                        help="random seed (default: 42)")
    parser.add_argument("--log-path", type=str, default="",
                        help="Path to save logs. Print to StdOut if log-path is not set")
    parser.add_argument("--save-model", action="store_true", default=False,
                        help="For Saving the current Model")

    args = parser.parse_args()

    use_cuda = not args.no_cuda and torch.cuda.is_available()
    if use_cuda:
        print("Using CUDA")

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    # Use this format (%Y-%m-%dT%H:%M:%SZ) to record timestamp of the metrics.
    # If log_path is empty print log to StdOut, otherwise print log to the file.
    if args.log_path == "":
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.INFO)
    else:
        logging.basicConfig(
            format="%(asctime)s %(levelname)-8s %(message)s",
            datefmt="%Y-%m-%dT%H:%M:%SZ",
            level=logging.INFO,
            filename=args.log_path)

    train_dl, test_dl = get_dataloaders(DATASET_DIR, args.batch_size, args.num_workers)

    # I use the wonderful timm package to load the pretrained efficientnet model
    # https://github.com/rwightman/pytorch-image-models
    model_ft = timm.create_model(MODEL_NAME, pretrained=True)

    # replace the last fc layer with an untrained one (requires grad by default)
    num_ftrs = model_ft.classifier.in_features
    model_ft.classifier = nn.Linear(num_ftrs, NUM_CAR_CLASSES)
    model_ft.to(device)

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model_ft.parameters(), lr=args.lr)

    lrscheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer,
                                                       mode='max',
                                                       patience=3,
                                                       threshold=0.9,
                                                       min_lr=1e-6,
                                                       verbose=True,
                                                       )

    model_ft, training_losses, training_accs, test_losses, test_accs = train_model(model_ft,
                                                                                   device,
                                                                                   train_dl,
                                                                                   test_dl,
                                                                                   loss_fn,
                                                                                   optimizer,
                                                                                   lrscheduler,
                                                                                   args.save_model,
                                                                                   n_epochs=args.epochs,
                                                                                   )


if __name__ == "__main__":
    main()