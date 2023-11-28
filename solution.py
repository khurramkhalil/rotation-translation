import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
from torchvision import transforms
from torchvision.datasets import VOCSegmentation
from torch.utils.data import DataLoader
import torchvision.models as models

import wandb

# Initialize Weights & Biases for experiment tracking
wandb.init(project="PASCAL-VOC2012-Sementic-Seg", name="sementic-segmentation")

# Check for GPU availability
device = 'cuda' if torch.cuda.is_available() else 'cpu'


def iou(preds, labels, num_classes=21):
    """
    Calculate Intersection over Union (IoU) for semantic segmentation.

    Args:
        preds (torch.Tensor): Predicted segmentation.
        labels (torch.Tensor): Ground truth segmentation.
        num_classes (int): Number of classes.

    Returns:
        float: Mean IoU score.
    """
    ious = torch.zeros(num_classes)
    preds = torch.argmax(preds, dim=1)
    label_and_pred = torch.stack((labels, preds), dim=2)

    for cls in range(num_classes):
        pred_inds = preds == cls
        label_inds = labels == cls
        intersection = (label_inds & pred_inds).long().sum().float()
        union = (label_inds | pred_inds).long().sum().float()

        if union == 0:
            ious[cls] = 1
        else:
            ious[cls] = intersection / union

    return torch.mean(ious).item()


def train(model, train_loader, optimizer, epoch, criterion, device):
    """
    Train the semantic segmentation model.

    Args:
        model (torch.nn.Module): Semantic segmentation model.
        train_loader (torch.utils.data.DataLoader): Training data loader.
        optimizer (torch.optim.Optimizer): Model optimizer.
        epoch (int): Current epoch number.
        criterion (torch.nn.Module): Loss function.
        device (str): Device ('cuda' or 'cpu').

    Returns:
        None
    """
    model.train()

    train_loss = 0.0
    train_iou = 0.0

    num_batches = len(train_loader)

    with tqdm(total=num_batches, desc=f'Epoch {epoch}', unit='batch') as t:
        for images, masks in train_loader:

            images = images.to(device)
            masks = masks.to(device)

            optimizer.zero_grad()

            output = model(images)

            # Calculate CrossEntropyLoss
            loss = criterion(output['out'], masks.squeeze(1).long())

            # Backward pass and optimization
            loss.backward()
            optimizer.step()

            # Update metrics
            train_loss += loss.item()
            iou_batch = iou(output['out'], masks.squeeze(1))
            train_iou += iou_batch

            t.set_postfix(loss=train_loss / (t.n + 1), IoU=train_iou / (t.n + 1))
            t.update()

    train_loss /= num_batches
    train_iou /= num_batches

    print(f'Epoch {epoch}, Train Loss: {train_loss:.4f}, Train IoU: {train_iou:.4f}')

    return train_loss, train_iou


def validate(model, val_loader, criterion, device):
    """
    Validate the semantic segmentation model.

    Args:
        model (torch.nn.Module): Semantic segmentation model.
        val_loader (torch.utils.data.DataLoader): Validation data loader.
        criterion (torch.nn.Module): Loss function.
        device (str): Device ('cuda' or 'cpu').

    Returns:
        dict: Validation metrics (loss and IoU).
    """
    model.eval()

    val_loss = 0.0
    val_iou = 0.0

    num_batches = len(val_loader)

    with torch.no_grad():
        with tqdm(total=num_batches, desc='Validation', unit='batch') as t:
            for images, masks in val_loader:
                
                images = images.to(device)
                masks = masks.to(device)

                output = model(images)

                # Calculate CrossEntropyLoss
                loss = criterion(output['out'], masks.squeeze(1).long())

                # Update metrics
                val_loss += loss.item()
                iou_batch = iou(output['out'], masks.squeeze(1))
                val_iou += iou_batch

                t.set_postfix(loss=val_loss / (t.n + 1), IoU=val_iou / (t.n + 1))
                t.update()

    val_loss /= num_batches
    val_iou /= num_batches

    print(f'Validation Loss: {val_loss:.4f}, Validation IoU: {val_iou:.4f}')
    return {'loss': val_loss, 'iou': val_iou}


def get_example_images(model, test_loader, device, num_examples=3):
    """
    Get a few example images, ground truth masks, and predicted masks from the test dataset.

    Args:
        model (torch.nn.Module): Semantic segmentation model.
        test_loader (torch.utils.data.DataLoader): Test data loader.
        device (str): Device ('cuda' or 'cpu').
        num_examples (int): Number of examples to retrieve.

    Returns:
        Tuple[List[torch.Tensor], List[torch.Tensor], List[torch.Tensor]]:
            Tuple containing lists of example images, ground truth masks, and predicted masks.
    """
    model.eval()

    example_images = []
    example_masks = []
    example_preds = []

    with torch.no_grad():
        for i, (images, masks) in enumerate(test_loader):
            if i >= 1:
                break
            
            for j in range(images.shape[0]):
                if j >= 3:
                    break
                image = images[j, :, :].unsqueeze(0).to(device)
                mask = masks[j, :, :].unsqueeze(0).to(device)

                output = model(image)

                # Assuming output contains the predicted mask (modify accordingly)
                predicted_mask = torch.argmax(output['out'], dim=1).cpu()

                example_images.append(image.cpu())
                example_masks.append(mask.cpu())
                example_preds.append(predicted_mask)

    return example_images, example_masks, example_preds


def overlay(image, mask, color, alpha, resize=None):
    """Combines image and its segmentation mask into a single image.
    https://www.kaggle.com/code/purplejester/showing-samples-with-segmentation-mask-overlay

    Args:
        image: Training image. np.ndarray,
        mask: Segmentation mask. np.ndarray,
        color: Color for segmentation mask rendering.  tuple[int, int, int] = (255, 0, 0)
        alpha: Segmentation mask's transparency. float = 0.5,
        resize: If provided, both image and its mask are resized before blending them together.
        tuple[int, int] = (256, 256))

    Returns:
        image_combined: The combined image. np.ndarray

    """
    color = color[::-1]
    colored_mask = np.expand_dims(mask, 0).repeat(3, axis=0)
    colored_mask = np.moveaxis(colored_mask, 0, -1)
    masked = np.ma.MaskedArray(image, mask=colored_mask, fill_value=color)
    image_overlay = masked.filled()

    if resize is not None:
        image = cv2.resize(image.transpose(1, 2, 0), resize)
        image_overlay = cv2.resize(image_overlay.transpose(1, 2, 0), resize)

    image_combined = cv2.addWeighted(image, 1 - alpha, image_overlay, alpha, 0)

    return image_combined

# Define the transformations for training and testing
transform_train = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

transform_test = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
])

# Modify the dataset to directly return tensors
class PascalVOCDataset(VOCSegmentation):
    def __getitem__(self, index):
        image, target = super(PascalVOCDataset, self).__getitem__(index)

        # Convert the target (segmentation mask) to a numpy array
        target = np.array(target, dtype=np.uint8)

        # Convert the numpy array back to a PIL image
        target = Image.fromarray(target)

        # Apply transformations
        target = transforms.ToTensor()(target)
        target = transforms.Resize((256, 256))(target)
        image = transforms.Resize((256, 256))(image)

        return image, target

# Set the path to the Pascal VOC dataset
voc_root = "C:/Users/MSI/PycharmProjects/YOLO-v8/"
batch_size = 4

# Load the training dataset
train_dataset = PascalVOCDataset(root=voc_root, year='2012', image_set='train', download=False, transform=transform_train)

# Load the test dataset
test_dataset = PascalVOCDataset(root=voc_root, year='2012', image_set='val', download=False, transform=transform_test)

# Create data loaders for training and testing
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Define and instantiate the DeepLabV3 model with a ResNet-50 backbone for semantic segmentation
# - pretrained=False: Do not use pre-trained weights on ImageNet
# - weights_backbone='ResNet50_Weights.IMAGENET1K_V1': Use custom weights for the ResNet-50 backbone
# - progress=True: Display progress during the model download
# - num_classes=21: Number of classes for semantic segmentation (adjust based on your specific task)
model = models.segmentation.deeplabv3_resnet50(
    pretrained=False,
    weights_backbone='ResNet50_Weights.IMAGENET1K_V1',
    progress=True,
    num_classes=21
).to(device)

# Training parameters
num_epochs = 25

# Optimizer: Adam is used for parameter optimization
optimizer = torch.optim.Adam(model.parameters())

# Learning rate scheduler: Adjusts the learning rate during training
# - StepLR is used with a step size of 5 epochs and a gamma of 0.1
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# Loss criterion: CrossEntropyLoss is commonly used for semantic segmentation
criterion = nn.CrossEntropyLoss()


# Update Weights & Biases configuration
wandb.config.update({
    "num_epochs": num_epochs,
    "optimizer": optimizer.__class__.__name__,
    "learning_rate": optimizer.param_groups[0]["lr"],
    "scheduler": lr_scheduler.__class__.__name__,
    "step_size": lr_scheduler.step_size,
    "gamma": lr_scheduler.gamma,
    "criterion": criterion.__class__.__name__
})


# Training loop
for epoch in range(num_epochs):
    # Train the model on the training data
    train_loss, train_iou = train(model, train_loader, optimizer, epoch, criterion, device)
    
    # Validate the model on the test data and get validation metrics
    val_metrics = validate(model, test_loader, criterion, device)
    val_loss, val_iou = val_metrics['loss'], val_metrics['iou']

    # Get some example images and masks for logging
    example_images, example_masks, example_preds = get_example_images(model, test_loader, device, num_examples=3)

    # Log metrics to W&B
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": train_loss,
        "train_iou": train_iou,
        "val_loss": val_loss,
        "val_iou": val_iou
    })

    # Log images with overlays
    for i in range(len(example_images)):
        ground_truth_mask = example_masks[i].cpu().numpy()
        predicted_mask = example_preds[i].cpu().numpy()

        img = example_images[i][0].numpy()
        gt = ground_truth_mask[0]
        mask = predicted_mask

        image_gt = overlay(img.T, gt[0,:].T, color=(0,255,0), alpha=0.3)
        # Log the overlay image to W&B
        wandb.log({"Image with Ground Truth": wandb.Image(image_gt), "epoch": epoch + 1})

        image_predicted = overlay(img.T, mask[0,:].T, color=(0,255,0), alpha=0.3)
        wandb.log({"Image with Predicted Mask": wandb.Image(image_predicted), "epoch": epoch + 1})
    
    # Print epoch information and validation metrics
    print(f"Epoch {epoch+1}")
    print(val_metrics)
    
    # Adjust the learning rate based on the scheduler
    lr_scheduler.step()
