import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision import datasets
import torchvision.models as torch_models
import matplotlib.pyplot as plt
import cv2
import imgaug
import argparse

device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")

# batch_size = 256
# num_workers = 4
# lr = 1e-4
# epochs = 20
image_size = 28
data_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize(image_size),
    transforms.ToTensor()
])

opts = None

class FMDataset(Dataset):
    def __init__(self, df: pd.DataFrame, transform=None) -> None:
        super().__init__()
        self.df = df
        self.transform = transform
        self.images = df.iloc[:,1:].values.astype(np.uint8)
        self.labels = df.iloc[:, 0].values

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.images[idx].reshape(28, 28, 1)
        label = int(self.labels[idx])
        if self.transform is not None:
            image = self.transform(image)
        else:
            image = torch.tensor(image/255., dtype=torch.float)
        label = torch.tensor(label, dtype=torch.long)
        return image, label
    
class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 32, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
            nn.Conv2d(32, 64, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2),
            nn.Dropout(0.3),
        )
        self.fc = nn.Sequential(
            nn.Linear(64*4*4, 512),
            nn.ReLU(),
            nn.Linear(512, 10),
        )

    def forward(self: nn.Sequential, x):
        x = self.conv(x)
        x = x.view(-1, 64*4*4)
        x = self.fc(x)
        return x

def train(model: Net, train_loader: DataLoader, criterion: nn.CrossEntropyLoss, epoch):
    model.train()
    train_loss = 0
    optimizer = optim.Adam(model.parameters(), lr=opts.lr)
    for data, label in train_loader:
        # data, label = data.cuda(), label.cuda()
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, label)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * data.size(0)
    train_loss = train_loss / len(train_loader.dataset)
    print('Epoch: {} \tTraining Loss: {:.6f}'.format(epoch, train_loss))

def val(model: Net, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, epoch):
    model.eval()
    val_loss = 0
    gt_labels = []
    pred_labels = []
    with torch.no_grad():
        for data, label in test_loader:
            # data, label = data.cuda(), label.cuda()
            output = model(data)
            preds = torch.argmax(output, 1)
            gt_labels.append(label.cpu().data.numpy())
            pred_labels.append(preds.cpu().data.numpy())
            loss = criterion(output, label)
            val_loss += loss.item()*data.size(0)
    val_loss = val_loss/len(test_loader.dataset)
    gt_labels, pred_labels = np.concatenate(gt_labels), np.concatenate(pred_labels)
    acc = np.sum(gt_labels==pred_labels)/len(pred_labels)
    print('Epoch: {} \tValidation Loss: {:.6f}, Accuracy: {:6f}'.format(epoch, val_loss, acc))

def main():
    train_df = pd.read_csv('datasets/fashion-mnist_train.csv')
    test_df = pd.read_csv('datasets/fashion-mnist_test.csv')
    train_data = FMDataset(train_df, data_transform)
    test_data = FMDataset(test_df, data_transform)
    print('FMDataset loaded.')
    train_loader = DataLoader(train_data, batch_size=opts.batch_size, shuffle=True, num_workers=opts.workers, drop_last=True)
    test_loader = DataLoader(test_data, batch_size=opts.batch_size, shuffle=False, num_workers=opts.workers)
    print('DataLoader init.')

    image, label = next(iter(train_loader))
    print(image.shape, label.shape)
    # print(f"type of image: {type(image)}")
    # plt.imshow(image[1][0], cmap='gray')
    # plt.savefig('image1.jpg')

    model = Net()
    print(model)
    criterion = nn.CrossEntropyLoss()

    # model2 = torch_models.resnet50()
    # print(model2)

    for epoch in range(1, opts.epochs+1):
        train(model, train_loader, criterion, epoch)
        val(model, test_loader, criterion, epoch)
    torch.save(model, os.path.join(opts.output, "fashion_mnist_demo.pkl"))

def init_args():
    global opts
    parser = argparse.ArgumentParser()
    parser.add_argument('--workers', type=int, default=4, help='number of data loading workers')
    parser.add_argument('--batch_size', type=int, default=256, help='input batch size')
    parser.add_argument('--lr', type=float, default=3e-5, help='select learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='num of epochs to train')
    parser.add_argument('--checkpoint_path', type=str, default='', help='path to load previous trained model if not empty')
    parser.add_argument('--cuda', action='store_true', default=False, help='enable cuda')
    parser.add_argument('--output', type=str, default='', help='output model path')
    opts = parser.parse_args()

if __name__ == '__main__':
    init_args()
    main()
