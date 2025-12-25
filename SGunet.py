#!/usr/bin/env python
#This code was developed by Raana Naanjam, a PhD student in Remote Sensing,

import os, cv2
import numpy as np
import pandas as pd
import random, tqdm
import seaborn as sns
import matplotlib.pyplot as plt
import warnings
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import albumentations as album
import segmentation_models_pytorch as smp
import os
from models.sgu_net import SVMGuidedAttentionUNet

DATA_DIR = 'proposedunet'

x_train_dir = os.path.join(DATA_DIR, 'train', 'image')
y_train_dir = os.path.join(DATA_DIR, 'train', 'label')
svm_train_dir = os.path.join(DATA_DIR, 'train', 'svm')  

x_valid_dir = os.path.join(DATA_DIR, 'val', 'image')
y_valid_dir = os.path.join(DATA_DIR, 'val', 'label')
svm_valid_dir = os.path.join(DATA_DIR, 'val', 'svm') 

x_test_dir = os.path.join(DATA_DIR, 'test', 'image')
y_test_dir = os.path.join(DATA_DIR, 'test', 'label')
svm_test_dir = os.path.join(DATA_DIR, 'test', 'svm')  

class_names = ['background', 'building']


class_rgb_values = [
    [0, 0, 0],       # background
    [255, 255, 255]  # building
]

print('Dataset classes and their corresponding RGB values:')
print('Class Names:', class_names)
print('Class RGB values:', class_rgb_values)


# helper function for data visualization
def visualize(**images):
    """
    Plot images in one row
    """
    n_images = len(images)
    plt.figure(figsize=(20,8))
    for idx, (name, image) in enumerate(images.items()):
        plt.subplot(1, n_images, idx + 1)
        plt.xticks([]); 
        plt.yticks([])
        # get title from the parameter names
        plt.title(name.replace('_',' ').title(), fontsize=20)
        plt.imshow(image)
    plt.show()

# Perform one hot encoding on label
def one_hot_encode(label, label_values):
    """
    Convert a segmentation image label array to one-hot format
    by replacing each pixel value with a vector of length num_classes
    # Arguments
        label: The 2D array segmentation image label
        label_values
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of num_classes
    """
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis = -1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)

    return semantic_map
    
# Perform reverse one-hot-encoding on labels / preds
def reverse_one_hot(image):
    """
    Transform a 2D array in one-hot format (depth is num_classes),
    to a 2D array with only 1 channel, where each pixel value is
    the classified class key.
    # Arguments
        image: The one-hot format image 
        
    # Returns
        A 2D array with the same width and hieght as the input, but
        with a depth size of 1, where each pixel value is the classified 
        class key.
    """
    x = np.argmax(image, axis = -1)
    return x

# Perform colour coding on the reverse-one-hot outputs
def colour_code_segmentation(image, label_values):
    """
    Given a 1-channel array of class keys, colour code the segmentation results.
    # Arguments
        image: single channel array where each value represents the class key.
        label_values

    # Returns
        Colour coded image for segmentation visualization
    """
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]

    return x


import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

def binary_mask_to_rgb(mask):
    rgb_mask = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
    rgb_mask[mask == 1] = [255, 255, 255]  
    rgb_mask[mask == 0] = [0, 0, 0]         
    return rgb_mask

class BuildingsDataset(torch.utils.data.Dataset):
    def __init__(
            self, 
            images_dir, 
            masks_dir, 
            class_rgb_values=None, 
            svm_dir=None,
            augmentation=None, 
            preprocessing=None,
    ):
        self.image_paths = [os.path.join(images_dir, image_id) for image_id in sorted(os.listdir(images_dir))]
        self.mask_paths = [os.path.join(masks_dir, image_id) for image_id in sorted(os.listdir(masks_dir))]
        self.class_rgb_values = class_rgb_values
        self.svm_dir = svm_dir
        self.augmentation = augmentation
        self.preprocessing = preprocessing

    def __getitem__(self, i):
        # خواندن تصویر RGB
        image = cv2.cvtColor(cv2.imread(self.image_paths[i]), cv2.COLOR_BGR2RGB)

   
        mask_gray = cv2.imread(self.mask_paths[i], cv2.IMREAD_GRAYSCALE)
        mask_binary = (mask_gray > 127).astype(np.uint8)
        mask_rgb = binary_mask_to_rgb(mask_binary)
        mask = one_hot_encode(mask_rgb, self.class_rgb_values).astype('float')

        
        svm_map = None
        if self.svm_dir is not None:
            image_filename = os.path.basename(self.image_paths[i])
            svm_filename = os.path.splitext(image_filename)[0] + ".tif"
            svm_path = os.path.join(self.svm_dir, svm_filename)

            if os.path.exists(svm_path):
                svm_raw = cv2.imread(svm_path)
                svm_gray = cv2.cvtColor(svm_raw, cv2.COLOR_BGR2GRAY)
                svm_map = svm_gray.astype('float32') / 255.0  # نرمال‌سازی

             
                if svm_map.shape != image.shape[:2]:
                    svm_map = cv2.resize(svm_map, (image.shape[1], image.shape[0]), interpolation=cv2.INTER_LINEAR)

     
        if self.augmentation:
            if svm_map is not None:
                sample = self.augmentation(image=image, mask=mask, masks=[svm_map])
                image, mask, svm_map = sample['image'], sample['mask'], sample['masks'][0]
            else:
                sample = self.augmentation(image=image, mask=mask)
                image, mask = sample['image'], sample['mask']

    
        if self.preprocessing:
            sample = self.preprocessing(image=image, mask=mask)
            image, mask = sample['image'], sample['mask']

        
        if svm_map is not None:
            svm_map = np.expand_dims(svm_map, axis=0)  # [1, H, W]
            svm_tensor = torch.tensor(svm_map, dtype=torch.float)
            return image, mask, svm_tensor

        return image, mask

    def __len__(self):
        return len(self.image_paths)


dataset = BuildingsDataset(x_train_dir, y_train_dir, class_rgb_values=select_class_rgb_values, svm_dir=svm_train_dir)

random_idx = random.randint(0, len(dataset)-1)
output = dataset[random_idx]

if len(output) == 3:
    image, mask, svm_map = output
else:
    image, mask = output
    svm_map = None


visualize(
    original_image = image,
    ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
    one_hot_encoded_mask = reverse_one_hot(mask),
    svm_probability_map = svm_map[0]# if svm_map is not None else np.zeros_like(mask[..., 0])  # فقط کانال ۱‌ـی SVM
)


def get_training_augmentation():
    train_transform = [    
        album.RandomCrop(height=256, width=256, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)


def get_validation_augmentation():   
   
    test_transform = [
        album.PadIfNeeded(min_height=512, min_width=512, always_apply=True, border_mode=0),
    ]
    return album.Compose(test_transform)


def to_tensor(x, **kwargs):
    return x.transpose(2, 0, 1).astype('float32')


def get_preprocessing(preprocessing_fn=None):
    """Construct preprocessing transform    
    Args:
        preprocessing_fn (callable): data normalization function 
            (can be specific for each pretrained neural network)
    Return:
        transform: albumentations.Compose
    """   
    _transform = []
    if preprocessing_fn:
        _transform.append(album.Lambda(image=preprocessing_fn))
    _transform.append(album.Lambda(image=to_tensor, mask=to_tensor))
        
    return album.Compose(_transform)



augmented_dataset = BuildingsDataset(
    x_train_dir,
    y_train_dir, 
    svm_dir=svm_train_dir,  
    augmentation=get_training_augmentation(),
    class_rgb_values=select_class_rgb_values,
)

random_idx = random.randint(0, len(augmented_dataset) - 1)

for i in range(3):
    output = augmented_dataset[random_idx]

    if len(output) == 3:
        image, mask, svm_map = output
    else:
        image, mask = output
        svm_map = None


    image_vis = (image * 255).astype(np.uint8) if image.max() <= 1 else image.astype(np.uint8)

    visualize(
        original_image = image_vis,
        ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
        one_hot_encoded_mask = reverse_one_hot(mask),
        svm_probability_map = svm_map.squeeze(0) if svm_map is not None else np.zeros((image.shape[1], image.shape[2]))
    )



ENCODER = 'resnet50'
ENCODER_WEIGHTS = 'imagenet'
CLASSES = class_names
ACTIVATION = 'sigmoid' 
model = SVMGuidedAttentionUNet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)
preprocessing_fn = smp.encoders.get_preprocessing_fn(ENCODER, ENCODER_WEIGHTS)


train_dataset = BuildingsDataset(
    x_train_dir,
    y_train_dir, 
    svm_dir=svm_train_dir,
    augmentation=get_training_augmentation(),
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)


valid_dataset = BuildingsDataset(
    x_valid_dir,
    y_valid_dir, 
    svm_dir=svm_valid_dir,  
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

# DataLoaders
train_loader = DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=0)
valid_loader = DataLoader(valid_dataset, batch_size=1, shuffle=False, num_workers=0)



import os
import torch
import torch.nn.functional as F
from torchmetrics.classification import JaccardIndex

TRAINING = True
EPOCHS = 100
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SVMGuidedAttentionUNet(
    encoder_name=ENCODER, 
    encoder_weights=ENCODER_WEIGHTS, 
    classes=len(CLASSES), 
    activation=ACTIVATION,
)

model_path = "saved_model_path/best_model.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    print("✅ Model weights loaded.")

model.to(DEVICE)


def weighted_bce_loss(pred, target, svm_weight):
    bce = F.binary_cross_entropy(pred[:, 1], target[:, 1], reduction='none')
    return (bce * svm_weight.squeeze(1)).mean()


metrics_fn = JaccardIndex(task="binary")


optimizer = torch.optim.Adam([
    dict(params=model.parameters(), lr=1e-4),
])


lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=1, T_mult=2, eta_min=5e-5,
)



import torch.nn as nn


def dice_loss(pred, target, smooth=1e-6):
    pred = pred.contiguous()
    target = target.contiguous()
    
    intersection = (pred * target).sum(dim=(2, 3))
    union = pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3))
    
    dice = (2. * intersection + smooth) / (union + smooth)
    return 1 - dice.mean()


bce_loss = nn.BCEWithLogitsLoss()


def boundary_loss(pred, target):
   
    laplace_filter = torch.tensor([[0, 1, 0], [1, -4, 1], [0, 1, 0]], dtype=torch.float32, device=pred.device).unsqueeze(0).unsqueeze(0)
    pred_edges = F.conv2d(pred, laplace_filter, padding=1)
    target_edges = F.conv2d(target, laplace_filter, padding=1)
    return F.l1_loss(pred_edges, target_edges)



from models.sgu_Loss import combined_loss


import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(DEVICE)

def train_epoch(model, data_loader, loss_fn, optimizer, device):
    model.train()
    total_loss = 0

    for batch in tqdm(data_loader, desc="Training"):
        images, masks, svm_maps = unpack_batch(batch, device)
        
        preds = model(images, svm_map=svm_maps)
        loss = loss_fn(preds[:, 1:2], masks[:, 1:2], svm_maps)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    avg = total_loss / len(data_loader)
    print(f"✅ Training Loss: {avg:.4f}")
    return avg

# ------------------ Validation ------------------
def validate_epoch(model, data_loader, loss_fn, metric_fn, device):
    model.eval()
    total_loss, total_metric = 0, 0

    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Validation"):
            images, masks, svm_maps = unpack_batch(batch, device)

            preds = model(images, svm_map=svm_maps)
            loss = loss_fn(preds[:, 1:2], masks[:, 1:2], svm_maps)

            total_loss += loss.item()
            total_metric += metric_fn((preds[:, 1:2] > 0.5).float(), masks[:, 1:2]).item()

    avg_loss = total_loss / len(data_loader)
    avg_metric = total_metric / len(data_loader)
    print(f"📊 Val Loss: {avg_loss:.4f}, Val IoU: {avg_metric:.4f}")
    return avg_loss, avg_metric

# ------------------ Jaccard ------------------
def jaccard_index(pred, target, smooth=1e-10):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum() - intersection
    return (intersection + smooth) / (union + smooth)

# ------------------ Helper ------------------
def unpack_batch(batch, device):
    if len(batch) == 3:
        images, masks, svm_maps = batch
        return images.to(device), masks.to(device), svm_maps.to(device)
    else:
        images, masks = batch
        return images.to(device), masks.to(device), None

# ------------------ Training Loop ------------------
best_metric = 0.0
save_dir = "./"
best_model_state_path = os.path.join(save_dir, "best_modelsvm_state.pth")
best_model_full_path = os.path.join(save_dir, "best_model_fullsvm.pth")

for epoch in range(EPOCHS):
    print(f"\n🌟 Epoch {epoch + 1}/{EPOCHS}")

    train_loss = train_epoch(
        model=model,
        data_loader=train_loader,
        loss_fn=combined_loss,
        optimizer=optimizer,
        device=DEVICE,
    )

    val_loss, val_metric = validate_epoch(
        model=model,
        data_loader=valid_loader,
        loss_fn=combined_loss,
        metric_fn=jaccard_index,
        device=DEVICE,
    )

    if val_metric > best_metric:
        best_metric = val_metric
        torch.save(model.state_dict(), best_model_state_path)
        torch.save(model, best_model_full_path)
        print(f"💾 Best model saved — Val IoU: {best_metric:.4f}")

epochs_range = range(1, EPOCHS+1)

plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.plot(epochs_range, train_losses, label="Training Loss")
plt.plot(epochs_range, val_losses, label="Validation Loss")
plt.title("Loss per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()


plt.subplot(1, 2, 2)
plt.plot(epochs_range, val_metrics, label="Validation Metric")
plt.title("Validation Metric per Epoch")
plt.xlabel("Epochs")
plt.ylabel("Metric (Jaccard Index)")
plt.legend()

plt.tight_layout()
plt.show()



import os
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.exists('./best_modelsvm_state.pth'):
    print("✅ Found best_model_state.pth! Loading state_dict into model...")

    best_model =  SVMGuidedAttentionUNet(
        encoder_name='resnet50',
        encoder_weights=None,
        classes=2,
        activation='sigmoid'
    ).to(DEVICE)

    best_model.load_state_dict(torch.load('./best_modelsvm_state.pth', map_location=DEVICE))
    best_model.eval()


elif os.path.exists('./best_model_fullsvm.pth'):
    print("✅ Found best_model_full.pth! Loading full model...")

    best_model = torch.load('./best_model_full.pth', map_location=DEVICE)
    best_model = best_model.to(DEVICE)
    best_model.eval()

else:
    print("❌ No saved model found!")


test_dataset = BuildingsDataset(
    x_test_dir, 
    y_test_dir, 
    svm_dir=svm_test_dir, 
    augmentation=get_validation_augmentation(), 
    preprocessing=get_preprocessing(preprocessing_fn),
    class_rgb_values=select_class_rgb_values,
)

test_dataloader = DataLoader(test_dataset)


test_dataset_vis = BuildingsDataset(
    x_test_dir, 
    y_test_dir, 
    augmentation=get_validation_augmentation(),
    class_rgb_values=select_class_rgb_values,
)


random_idx = random.randint(0, len(test_dataset_vis)-1)
image, mask = test_dataset_vis[random_idx]

visualize(
    original_image = image,
    ground_truth_mask = colour_code_segmentation(reverse_one_hot(mask), select_class_rgb_values),
    one_hot_encoded_mask = reverse_one_hot(mask)
)


def crop_image(image, target_image_dims=[5000, 5000, 3]):
    target_height, target_width = target_image_dims[:2]
    height, width = image.shape[:2]

    pad_top = (height - target_height) // 2
    pad_left = (width - target_width) // 2

    return image[
        pad_top : pad_top + target_height,
        pad_left : pad_left + target_width,
        :,
    ]



sample_preds_folder = 'sample_predictions/'
if not os.path.exists(sample_preds_folder):
    os.makedirs(sample_preds_folder)


import torch.nn.functional as F
import math

def pad_to_divisible(image_tensor, divisor=32):
    """Pads tensor so that height and width are divisible by `divisor`"""
    _, _, h, w = image_tensor.shape
    pad_h = math.ceil(h / divisor) * divisor - h
    pad_w = math.ceil(w / divisor) * divisor - w
    padded = F.pad(image_tensor, (0, pad_w, 0, pad_h))  # (left, right, top, bottom)
    return padded, (pad_h, pad_w)



for idx in range(len(test_dataset)):


    sample = test_dataset[idx]
    
   
    if len(sample) == 3:
        image, gt_mask, svm_map = sample
        svm_tensor = torch.unsqueeze(svm_map, 0).to(DEVICE)  # [1, 1, H, W]
    else:
        image, gt_mask = sample
        svm_tensor = None

   
    image_vis = crop_image(test_dataset_vis[idx][0].astype('uint8'))

   
    x_tensor = torch.from_numpy(image).unsqueeze(0).to(DEVICE)  # [1, C, H, W]

   
    x_tensor, (pad_h, pad_w) = pad_to_divisible(x_tensor, divisor=32)
    if svm_tensor is not None:
        svm_tensor, _ = pad_to_divisible(svm_tensor, divisor=32)

    
    with torch.no_grad():
        if svm_tensor is not None:
            pred_mask = best_model(x_tensor, svm_map=svm_tensor)
        else:
            pred_mask = best_model(x_tensor)


    if pad_h > 0 or pad_w > 0:
        pred_mask = pred_mask[:, :, :-pad_h if pad_h > 0 else None, :-pad_w if pad_w > 0 else None]

    pred_mask = pred_mask.squeeze().cpu().numpy()
    pred_mask = np.transpose(pred_mask, (1, 2, 0))  # CHW → HWC


    pred_building_heatmap = pred_mask[:, :, select_classes.index('building')]

  
    pred_mask_rgb = crop_image(colour_code_segmentation(reverse_one_hot(pred_mask), select_class_rgb_values))

   
    gt_mask = np.transpose(gt_mask, (1, 2, 0))
    gt_mask_rgb = crop_image(colour_code_segmentation(reverse_one_hot(gt_mask), select_class_rgb_values))

  
    cv2.imwrite(
        os.path.join(sample_preds_folder, f"sample_pred_{idx}.png"),
        np.hstack([image_vis, gt_mask_rgb, pred_mask_rgb])[:, :, ::-1]  # BGR برای OpenCV
    )


    visualize(
        original_image=image_vis,
        ground_truth_mask=gt_mask_rgb,
        predicted_mask=pred_mask_rgb,
        predicted_building_heatmap=pred_building_heatmap
    )

