import cv2
import os
import pickle
import numpy as np

import json

import torch
from torch import optim
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
from torchvision.transforms import Compose, ToTensor, Resize

from resnet import ResNet18
from xception import Xception
from attention_unet import AttentionUNet
from resnet50_bb import ResNet50ForBoundingBox
from maskRCNN import MaskRCNN

from sklearn.model_selection import train_test_split


class CustomDataset(Dataset):
    def __init__(self, image_dir, mask_dir, transform=None):
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = Compose([ToTensor()])
        self.images = os.listdir(image_dir)
        self.images.sort()
        self.masks = os.listdir(mask_dir)
        self.masks.sort()

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        box_path = os.path.join(self.mask_dir, self.masks[idx])

        with open(box_path, 'r') as f:
            data = f.readline().split()
            box = [float(x) for x in data]


        # Read image and mask using OpenCV
        image = cv2.imread(img_path)
        # box[0] /= 320
        # box[1] /= 285
        # box[2] /= 320
        # box[3] /= 285
        #
        # box[0] *= 256
        # box[1] *= 256
        # box[2] *= 256
        # box[3] *= 256

        # Convert from BGR to RGB
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        size = 256
        image = cv2.resize(image, (size, size))

        image = self.transform(image)

        # if self.transform:
        #     augmented = self.transform(image=image, mask=mask)
        #     image = augmented['image']
        #     mask = augmented['mask']

        image = image.type(torch.FloatTensor)

        return image, torch.tensor(box)

def compute_avg_iou(predictions, targets):
    """
    Compute average IoU for a batch of predictions and targets.

    Args:
    - predictions: tensor of shape [batch_size, 4] (x1, y1, x2, y2)
    - targets: tensor of shape [batch_size, 4] (x1, y1, x2, y2)

    Returns:
    - avg_iou: average IoU value
    """

    # Convert predictions and targets to bounding boxes [x1, y1, x2, y2]
    predictions = predictions.detach().cpu().numpy()
    targets = targets.detach().cpu().numpy()

    # Initialize IoU list
    iou_list = []

    # Compute IoU for each pair of prediction and target
    for pred, target in zip(predictions, targets):
        iou = compute_iou(pred, target)
        iou_list.append(iou)

    # Compute average IoU
    avg_iou = sum(iou_list) / len(iou_list)

    return avg_iou

def compute_iou(box1, box2):
    """
    Compute IoU between two bounding boxes.

    Args:
    - box1: list or tensor [x1, y1, x2, y2] (top-left corner and bottom-right corner)
    - box2: list or tensor [x1, y1, x2, y2] (top-left corner and bottom-right corner)

    Returns:
    - iou: IoU value
    """

    # Calculate intersection coordinates
    x1_intersect = max(box1[0], box2[0])
    y1_intersect = max(box1[1], box2[1])
    x2_intersect = min(box1[2], box2[2])
    y2_intersect = min(box1[3], box2[3])

    # Calculate area of intersection
    intersection_area = max(0, x2_intersect - x1_intersect + 1) * max(0, y2_intersect - y1_intersect + 1)

    # Calculate area of boxes
    box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
    box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)

    # Calculate union area
    union_area = box1_area + box2_area - intersection_area

    # Calculate IoU
    iou = intersection_area / union_area

    return iou

def calculate_iou(box1, box2):
    iou = 0.0
    for i in range(len(box1[0])):
        x1 = max(box1[i, 0], box2[i, 0])
        y1 = max(box1[i, 1], box2[i, 1])
        x2 = min(box1[i, 2], box2[i, 2])
        y2 = min(box1[i, 3], box2[i, 3])

        intersection_area = max(0, x2 - x1 + 1) * max(0, y2 - y1 + 1)
        box1_area = (box1[i, 2] - box1[i, 0] + 1) * (box1[i, 3] - box1[i, 1] + 1)
        box2_area = (box2[i, 2] - box2[i, 0] + 1) * (box2[i, 3] - box2[i, 1] + 1)

        iou += intersection_area / float(box1_area + box2_area - intersection_area)
    return iou


def calculate_accuracy(ground_truth, predictions, threshold=0.5):
    TP = 0
    FP = 0
    FN = 0

    for pred_box in predictions:
        max_iou = 0
        for gt_box in ground_truth:
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
        if max_iou >= threshold:
            TP += 1
        else:
            FP += 1

    for gt_box in ground_truth:
        max_iou = 0
        for pred_box in predictions:
            iou = calculate_iou(pred_box, gt_box)
            if iou > max_iou:
                max_iou = iou
        if max_iou < threshold:
            FN += 1

    precision = TP / (TP + FP) if TP + FP > 0 else 0
    recall = TP / (TP + FN) if TP + FN > 0 else 0
    accuracy = TP / (TP + FP + FN) if TP + FP + FN > 0 else 0

    return precision, recall, accuracy


NUM_OF_CLASSES = 1
LEARNING_RATE = 0.00025
NUM_EPOCHS = 100
WEIGHT_DECAY = 10e-5
BATCH_SIZE = 16

if __name__ == '__main__':

    # Define paths to your images and masks
    image_dir = 'dataSet/images'
    mask_dir = 'dataSet/masks'

    # Create dataset
    dataset = CustomDataset(image_dir, mask_dir)

    # Split dataset
    train_size = 0.7
    val_size = 0.15
    test_size = 0.15

    train_dataset, temp_dataset = train_test_split(dataset, test_size=1 - train_size)
    val_dataset, test_dataset = train_test_split(temp_dataset, test_size=test_size/(test_size + val_size))

    # Create DataLoaders
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(device)
    # model = UNet(3, 1).to(device)
    # model = UNet(n_class=NUM_OF_CLASSES).to(device)
    model = ResNet18().to(device)
    # model = Xception().to(device)
    # model = MaskRCNN().to(device)
    # model = ResNet50ForBoundingBox().to(device)
    criterion = nn.MSELoss().to(device) # or your chosen loss function
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

    history = {'train_loss': [], 'train_iou': [], 'val_loss': [], 'val_iou': [], 'test_loss': [], 'test_iou': []}

    num_epochs = NUM_EPOCHS

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        train_loss = 0.0

        # During training loop
        total_iou_train = 0.0
        total_batches_train = 0

        for batch in train_loader:  # Assuming you have DataLoader for train set
            inputs, targets = batch
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()  # Zero the parameter gradients

            outputs = model(inputs)  # Forward pass
            loss = criterion(outputs, targets)  # Compute loss
            loss.backward()  # Backward pass
            # torch.nn.utils.clip_grad_norm_(model.parameters(), clip_value)
            optimizer.step()  # Optimize

            train_loss += loss.item() * inputs.size(0)
            # Compute IoU on training set
            with torch.no_grad():
                avg_iou_batch = compute_avg_iou(outputs, targets)
                total_iou_train += avg_iou_batch
                total_batches_train += 1

        # Calculate average losses
        train_loss = train_loss / len(train_loader.dataset)
        avg_iou_train = total_iou_train / total_batches_train

        print(f"Epoch {epoch+1}/{num_epochs}, Training Loss: {train_loss:.4f}")
        history['train_loss'].append(train_loss)
        print(f"Training IoU: {avg_iou_train:.4f}")
        history['train_iou'].append(avg_iou_train)

        # Validation loop
        model.eval()
        val_loss = 0.0

        val_recall = 0.0
        val_precision = 0.0

        total_iou_val = 0.0
        total_batches_val = 0

        with torch.no_grad():
            for images, masks in val_loader:
                images, masks = images.to(device), masks.to(device)
                outputs = model(images)
                loss = criterion(outputs, masks)
                val_loss += loss.item() * images.size(0)
                avg_iou_batch = compute_avg_iou(outputs, masks)
                total_iou_val += avg_iou_batch
                total_batches_val += 1



        val_loss /= len(val_loader.dataset)

        print(f"Validation Loss: {val_loss:.4f}")
        history['val_loss'].append(val_loss)
        avg_iou_val = total_iou_val / total_batches_val
        print(f"Validation IoU: {avg_iou_val:.4f}")
        history['val_iou'].append(avg_iou_val)

        # Save model and history every 10 epochs
        if (epoch + 1) % 10 == 0:
            torch.save(model.state_dict(), f"resnet_epoch_{epoch + 1}.pth")
            with open(f"history_epoch_{epoch + 1}.pkl", 'wb') as f:
                pickle.dump(history, f)
    # Testing loop
    test_loss = 0.0

    total_iou_test = 0.0
    total_batches_test = 0

    with torch.no_grad():
        for images, masks in test_loader:
            images, masks = images.to(device), masks.to(device)
            outputs = model(images)
            loss = criterion(outputs, masks)
            test_loss += loss.item() * images.size(0)
            avg_iou_batch = compute_avg_iou(outputs, masks)
            total_iou_test += avg_iou_batch
            total_batches_test += 1


    test_loss /= len(test_loader.dataset)
    print(f"Test Loss: {test_loss:.4f}")

    history['test_loss'].append(test_loss)
    avg_iou_test = total_iou_val / total_batches_val
    print(f"Test IoU: {avg_iou_test:.4f}")
    history['test_loss'].append(avg_iou_test)

    # final save
    torch.save(model.state_dict(), "resnet_final.pth")
    with open("history_final.pkl", 'wb') as f:
        pickle.dump(history, f)

