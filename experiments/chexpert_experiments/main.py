import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as datasets
import dmfusion as fusion
import torchvision
import torchvision.transforms as transforms
from PIL import Image
import torchmetrics as metrics
import pandas as pd

class ChexpertDataset(datasets.Dataset):
    def __init__(self, df, transform=lambda *x: x):
        self.df = df
        self.transform = transform

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, item):
        row = self.df.iloc[item]

img_size = (3, 224, 224)
policy_size = 14
T = 20  # max number of forward diffusions
batch_size = 16
test_batch_size = 16
epochs = 100
lr = 1e-4


train_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomCrop(256),
    transforms.ToTensor()
])

test_transform = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(256),
    transforms.ToTensor()
])


train_df, test_df = None, None

train_data = ChexpertDataset(train_df, train_transform)
test_data = ChexpertDataset(test_df, test_transform)

train_loader = datasets.DataLoader(train_data, batch_size=batch_size, shuffle=True)
test_loader = datasets.DataLoader(test_data, batch_size=test_batch_size)

train_metrics = {"auc": metrics.AUROC("multilabel"), "accuracy": metrics.Accuracy("multilabel"),
                 "precision": metrics.Precision("multilabel"), "recall": metrics.Recall("multilabel"),
                 "f1": metrics.F1Score("multilabel"), "confusion_matrix": metrics.ConfusionMatrix("multilabel")}

test_metrics = {"auc": metrics.AUROC("multilabel"), "accuracy": metrics.Accuracy("multilabel"),
                "precision": metrics.Precision("multilabel"), "recall": metrics.Recall("multilabel"),
                "f1": metrics.F1Score("multilabel"), "confusion_matrix": metrics.ConfusionMatrix("multilabel")}

print_metrics = {"auc": True, "accuracy": True, "precision": True, "recall": True,
                 "f1": True, "confusion_matrix": False}


controller = fusion.ImageConditionedLinearUNet(img_size, policy_size, max_times=T,
                                               positional_features=18, conv_features=(8, 16, 32, 64, 128, 256),
                                               lin_features=(32, 64, 128, 256, 512, 1024))
hyperparameters = fusion.Hyperparameters(T, beta_start=0.0001, beta_end=0.2)
learner = fusion.StableDiffusionLearner(img_size, img_size, None, controller, hyperparameters)

optimizer = optim.Adam(learner.parameters(), lr=lr)

df_metrics = {"epoch": [], "train": [], "test": [], "loss": []}.update(dict(zip(print_metrics.keys(), [[] for _ in print_metrics])))

for epoch in range(epochs):
    # train segment
    run_loss = 0
    for i, (images, targets) in enumerate(train_loader):
        optimizer.zero_grad()

        # diffusion forward timesteps
        times = torch.randint(0, T, (batch_size,))
        loss = learner.get_losses(images, targets, times)
        loss.backward()

        optimizer.step()

        print("\r", f"epoch: {epoch}/{epochs}, iter {i}/{len(train_loader)}, avg_loss: {run_loss/(i+1)}")
    print()
    # get train metrics
    for i, (images, targets) in enumerate(train_loader):
        preds = learner.generate(images)
        for metric in train_metrics:
            train_metrics[metric].update(torch.sigmoid(preds), targets)

        prints = f"epoch: {epoch}/{epochs}, iter {i}/{len(train_loader)}"
        for metric in train_metrics:
            if print_metrics[metric]:
                prints += f", {metric}: {train_metrics[metric].compute()}"

        print("\r", prints)
    print()

    for i, (images, targets) in enumerate(test_loader):
        preds = learner.generate(images)
        for metric in test_metrics:
            test_metrics[metric].update(torch.sigmoid(preds), targets)

        prints = f"epoch: {epoch}/{epochs}, iter {i}/{len(test_loader)}"
        for metric in test_metrics:
            if print_metrics[metric]:
                prints += f", {metric}: {test_metrics[metric].compute()}"

        print("\r", prints)
    print()


