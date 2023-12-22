import torchvision.datasets

_exp_name = "sample"
# Import necessary packages.
import scipy.io
import numpy as np
import pandas as pd
import torch
import os
import torch.nn as nn
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import ConcatDataset, DataLoader, Subset, Dataset
from torchvision.datasets import DatasetFolder, VisionDataset
from tqdm.auto import tqdm
import random
import cv2

myseed = 1234  # set a random seed for reproducibility
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(myseed)
torch.manual_seed(myseed)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(myseed)

def rgb_to_hsv(rgb):

    r, g, b = rgb[0], rgb[1], rgb[2]
    maxc = torch.max(rgb, dim=0).values
    minc = torch.min(rgb, dim=0).values
    delta = maxc - minc

    # Hue calculation
    h = torch.zeros_like(maxc)
    mask = (maxc != minc)
    idx = (maxc == r) & mask
    h[idx] = ((g - b) / delta)[idx] % 6
    idx = (maxc == g) & mask
    h[idx] = ((b - r) / delta + 2)[idx]
    idx = (maxc == b) & mask
    h[idx] = ((r - g) / delta + 4)[idx]
    h = h / 6

    # Special case for a grey color (no color saturation)
    grey_mask = (maxc == minc)
    h[grey_mask] = 1

    # Saturation calculation
    s = torch.zeros_like(maxc)
    s[maxc != 0] = (delta / maxc)[maxc != 0]

    # Value calculation
    v = maxc

    hsv = torch.stack((h, s, v), dim=0)

    return hsv

def custom_hue_transform(im):

    im = np.array(im)
    im_rgb = im
   # im = cv2.cvtColor(im, cv2.COLOR_RGB2HSV)  # Change the picture from rgb to hsv color space
    im = Image.fromarray(np.uint8(im))
    toTensor = transforms.ToTensor()

    im = toTensor(im)

    im = rgb_to_hsv(im)

    im_rgb = toTensor(im_rgb)

    original_hue_channel = im[0, :, :]
    new_hue_channels = []
    for i in range(3):
        new_hue_distance_1 = abs(original_hue_channel - 1 / 3 * i)
        new_hue_distance_2 = torch.where(new_hue_distance_1 == 1, torch.tensor(1), 1 - new_hue_distance_1)
        condition_result = new_hue_distance_1 > 1 / 2
        hue_distance = torch.where(condition_result, new_hue_distance_2, new_hue_distance_1)
        new_hue_channel = 1 / 3 - hue_distance
        new_hue_channel = torch.clamp(new_hue_channel, min=0)
        new_hue_channel = new_hue_channel * 3
        new_hue_channels.append(new_hue_channel)
    new_hue_channels = torch.stack(new_hue_channels, dim=0)
    im_temp = torch.cat([ im_rgb, new_hue_channels], dim=0)

    return im_rgb

train_tfm = transforms.Compose([
    transforms.Resize((128, 128)),  # Resize the image into a fixed shape (height = width = 128)
    transforms.RandomHorizontalFlip(),  # Randomly flip images horizontally
    transforms.RandomRotation(degrees=30),  # Randomly rotate the image (up to 30 degrees)
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random affine transformation
    transforms.RandomResizedCrop(size=(128, 128), scale=(0.8, 1.0)),  # Random cropping and scaling
    custom_hue_transform,
])

test_tfm = transforms.Compose([
    transforms.Resize((128, 128)),
    custom_hue_transform,
])

class Classifier(nn.Module):
    def __init__(self):
        super(Classifier, self).__init__()
        # torch.nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)
        # torch.nn.MaxPool2d(kernel_size, stride, padding)
        # input dimension [8, 128, 128]
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, 3, 1, 1),  # [64, 128, 128]
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [64, 64, 64]

            nn.Conv2d(64, 128, 3, 1, 1), # [128, 64, 64]
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [128, 32, 32]

            nn.Conv2d(128, 256, 3, 1, 1), # [256, 32, 32]
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),      # [256, 16, 16]

            nn.Conv2d(256, 512, 3, 1, 1), # [512, 16, 16]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 8, 8]

            nn.Conv2d(512, 512, 3, 1, 1), # [512, 8, 8]
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(2, 2, 0),       # [512, 4, 4]
        )
        self.fc = nn.Sequential(
            nn.Linear(512*4*4, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 102)
        )

    def forward(self, x):
        out = self.cnn(x)
        out = out.view(out.size()[0], -1)
        return self.fc(out)


class CustomFlowersDataset(Dataset):
    def __init__(self, image_dir, labels_mat, indices, transform=test_tfm):
        self.image_dir = image_dir
        self.labels = scipy.io.loadmat(labels_mat)['labels'][0]
        self.indices = indices
        self.transform = transform

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        img_idx = self.indices[idx]
        img_path = self.image_dir + "/" + f"image_{img_idx:05d}.jpg"
        image = Image.open(img_path)
        image = self.transform(image)
        label = int(self.labels[img_idx-1]-1)
        return image, label




image_dir = 'dataset/flowers-102/jpg'
labels_mat = 'dataset/flowers-102/imagelabels.mat'

indices = np.arange(1, 8190 )
np.random.shuffle(indices)

num_train = int(0.6 * len(indices))
num_valid = int(0.2 * len(indices))

train_indices = indices[:num_train]
valid_indices = indices[num_train:num_train + num_valid]
test_indices = indices[num_train + num_valid:]

# "cuda" only when GPUs are available.
device = "cuda" if torch.cuda.is_available() else "cpu"

resmodel = torchvision.models.resnet50()
resmodel.fc = nn.Linear(2048,102)
resmodel.conv1 = torch.nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)

# Initialize a model, and put it on the device specified.
model = resmodel.to(device)


# The number of batch size.
batch_size = 64

# The number of training epochs.
n_epochs = 500

# If no improvement in 'patience' epochs, early stop.
patience = 80

# For the classification task, use cross-entropy as the measurement of performance.
criterion = nn.CrossEntropyLoss()

# Initialize optimizer, use adam, we can fine-tune some hyperparameters such as learning rate on our own.
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=1e-5)

# Construct train and valid datasets.
# The argument "loader" tells how torchvision reads the data.
'''
train_set = torchvision.dataset.(image_dir, labels_mat, train_indices, train_tfm)
valid_set = CustomFlowersDataset(image_dir, labels_mat, valid_indices, test_tfm)
'''
train_set = torchvision.datasets.Flowers102('dataset','test',train_tfm,download=True)
valid_set = torchvision.datasets.Flowers102('dataset','val',test_tfm,download=True)
train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
valid_loader = DataLoader(valid_set, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)


# Initialize trackers
stale = 0
best_acc = 0

for epoch in range(n_epochs):

    # ---------- Training ----------

    model.train()

    # These are used to record information in training.
    train_losses = []
    train_accs = []

    for batch in tqdm(train_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()
        #print(imgs.shape,labels.shape)

        # Forward the data. (Make sure data and model are on the same device.)
        logits = model(imgs.to(device))

        # Calculate the cross-entropy loss.
        # We don't need to apply softmax before computing cross-entropy as it is done automatically.
        loss = criterion(logits, labels.to(device))

        # Gradients stored in the parameters in the previous step should be cleared out first.
        optimizer.zero_grad()

        # Compute the gradients for parameters.
        loss.backward()

        # Clip the gradient norms for stable training.
        grad_norm = nn.utils.clip_grad_norm_(model.parameters(), max_norm=10)

        # Update the parameters with computed gradients.
        optimizer.step()

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        train_losses.append(loss.item())
        train_accs.append(acc)

    train_loss = sum(train_losses) / len(train_losses)
    train_acc = sum(train_accs) / len(train_accs)

    # Print the information.
    print(f"[ Train | {epoch + 1:03d}/{n_epochs:03d} ] loss = {train_loss:.5f}, acc = {train_acc:.5f}")

    # ---------- Validation ----------

    model.eval()

    # These are used to record information in validation.
    valid_loss = []
    valid_accs = []

    # Iterate the validation set by batches.
    for batch in tqdm(valid_loader):

        # A batch consists of image data and corresponding labels.
        imgs, labels = batch
        #imgs = imgs.half()

        with torch.no_grad():
            logits = model(imgs.to(device))

        loss = criterion(logits, labels.to(device))

        # Compute the accuracy for current batch.
        acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

        # Record the loss and accuracy.
        valid_loss.append(loss.item())
        valid_accs.append(acc)
        #break

    # The average loss and accuracy for entire validation set is the average of the recorded values.
    valid_loss = sum(valid_loss) / len(valid_loss)
    valid_acc = sum(valid_accs) / len(valid_accs)

    # Print the information.
    #print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # update logs
    if valid_acc > best_acc:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f} -> best")
    else:
        with open(f"./{_exp_name}_log.txt","a"):
            print(f"[ Valid | {epoch + 1:03d}/{n_epochs:03d} ] loss = {valid_loss:.5f}, acc = {valid_acc:.5f}")


    # save models
    if valid_acc > best_acc:
        print(f"Best model found at epoch {epoch}, saving model")
        torch.save(model.state_dict(), f"{_exp_name}_best1.ckpt") # only save best to prevent output memory exceed error
        best_acc = valid_acc
        stale = 0
    else:
        stale += 1
        if stale > patience:
            print(f"No improvment {patience} consecutive epochs, early stopping")
            break

# Construct test datasets.
# The argument "loader" tells how torchvision reads the data.
'''
test_set = CustomFlowersDataset(image_dir, labels_mat, test_indices, test_tfm)
'''
test_set = torchvision.datasets.Flowers102('dataset','train',test_tfm,download=True)
test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=0, pin_memory=True)

model_best = resmodel.to(device)
model_best.load_state_dict(torch.load(f"{_exp_name}_best1.ckpt"))
model_best.eval()
test_loss = []
test_accs = []
for batch in tqdm(test_loader):
    imgs, labels = batch
    #imgs = imgs.half()

    with torch.no_grad():
        logits = model_best(imgs.to(device))

    loss = criterion(logits, labels.to(device))

    # Compute the accuracy for current batch.
    acc = (logits.argmax(dim=-1) == labels.to(device)).float().mean()

    # Record the loss and accuracy.
    test_loss.append(loss.item())
    test_accs.append(acc)
    #break

# The average loss and accuracy for entire validation set is the average of the recorded values.
test_loss = sum(test_loss) / len(test_loss)
test_acc = sum(test_accs) / len(test_accs)

# Print the information.
print(f"[ test |  ] loss = {test_loss:.5f}, acc = {test_acc:.5f}")

"""
# create test csv
def pad4(i):
    return "0"*(4-len(str(i)))+str(i)
df = pd.DataFrame()
df["Id"] = [pad4(i) for i in range(len(test_set))]
df["Category"] = prediction
df.to_csv("submission.csv",index = False)
"""