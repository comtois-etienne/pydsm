# RESNET MODEL FOR TREE SPECIES CLASSIFICATION
# mostly written using the help of ChatGPT


from pathlib import Path

import numpy as np
import pandas as pd
import os
from PIL import Image

import torch
from torch import nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

from dataclasses import dataclass

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import pydsm.const as const
from .tile import Tile
from .tile import open_as_tile as tile_open_as_tile
from .nda import get_labels_centers as nda_get_labels_centers
from .nda import crop_using_mask as nda_crop_using_mask
from .utils import remove_extension as utils_remove_extension




# =============================================================================
# Configuration
# =============================================================================

TEST_RATIO = 0.20
BATCH_SIZE = 16
NUM_WORKERS = 0

LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4

RANDOM_SEED = 42



# =============================================================================
# Reproducibility
# =============================================================================

def set_seed(seed: int) -> None:
    """
    Set random seeds for reproducible training.

    :param seed: random seed.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# Image preprocessing
# =============================================================================

class MirrorPadResize:
    """
    Mirror-pad an image to a square and resize it.

    The original aspect ratio is preserved. The shorter dimension is extended
    by reflecting pixels along the image border.

    :param size: output image width and height.
    """

    def __init__(self, size: int):
        self.size = size

    def __call__(self, image: Image.Image) -> Image.Image:
        """
        Mirror-pad and resize an image.

        :param image: input PIL image.
        :return: square resized image.
        """
        image = image.convert('RGB')

        array = np.asarray(image)

        height, width = array.shape[:2]
        target_size = max(height, width)

        pad_height = target_size - height
        pad_width = target_size - width

        top = pad_height // 2
        bottom = pad_height - top

        left = pad_width // 2
        right = pad_width - left

        array = np.pad(
            array,
            (
                (top, bottom),
                (left, right),
                (0, 0),
            ),
            mode='reflect',
        )

        image = Image.fromarray(array)

        return image.resize(
            (self.size, self.size),
            Image.Resampling.BILINEAR,
        )


# =============================================================================
# Dataset
# =============================================================================

def create_transforms(imgs=512) -> tuple:
    """
    Create training and testing transformations.

    Training includes:
        - mirror padding
        - horizontal flip
        - vertical flip
        - random rotation
        - brightness adjustment
        - contrast adjustment
        - ImageNet normalization

    Testing includes only deterministic preprocessing and normalization.

    :return: training and testing transformations.
    """
    train_transform = transforms.Compose([
        # Preserve aspect ratio and fill the shorter dimension.
        MirrorPadResize(imgs),

        # Geometric augmentation.
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),

        transforms.RandomRotation(
            degrees=180,
            interpolation=transforms.InterpolationMode.BILINEAR,
        ),

        # Photometric augmentation.
        transforms.ColorJitter(
            brightness=0.20,
            contrast=0.20,
        ),

        transforms.ToTensor(),

        # Required for ImageNet-pretrained ResNet-50.
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_transform = transforms.Compose([
        MirrorPadResize(imgs),

        transforms.ToTensor(),

        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return train_transform, test_transform


def create_datasets(dataset_path: Path, classes_list: list[str], imgs=512) -> tuple:
    """
    Create stratified training and testing datasets using the class order
    defined by species_list.

    :param species_list: ordered list of class/folder names.
    :return: training dataset, testing dataset, class names.
    """
    train_transform, test_transform = create_transforms(imgs)

    # ImageFolder initially assigns alphabetical class indices.
    train_full = datasets.ImageFolder(
        root=dataset_path,
        transform=train_transform,
    )

    test_full = datasets.ImageFolder(
        root=dataset_path,
        transform=test_transform,
    )

    # -------------------------------------------------------------------------
    # Verify that the folders match species_list
    # -------------------------------------------------------------------------

    folder_classes = train_full.classes

    if set(folder_classes) != set(classes_list):
        missing = set(classes_list) - set(folder_classes)
        unexpected = set(folder_classes) - set(classes_list)

        raise ValueError(
            f'Class folders do not match species_list.\n'
            f'Missing folders: {sorted(missing)}\n'
            f'Unexpected folders: {sorted(unexpected)}'
        )

    if len(classes_list) != 16:
        raise ValueError(
            f'Expected 16 classes, found {len(classes_list)}.'
        )

    # -------------------------------------------------------------------------
    # Convert ImageFolder's alphabetical indices to species_list indices
    # -------------------------------------------------------------------------

    alphabetical_to_classes = {
        folder_index: classes_list.index(folder_name)
        for folder_index, folder_name in enumerate(folder_classes)
    }

    train_full.targets = [
        alphabetical_to_classes[label]
        for label in train_full.targets
    ]

    test_full.targets = [
        alphabetical_to_classes[label]
        for label in test_full.targets
    ]

    # ImageFolder samples also contain the original class index.
    train_full.samples = [
        (path, alphabetical_to_classes[label])
        for path, label in train_full.samples
    ]

    test_full.samples = [
        (path, alphabetical_to_classes[label])
        for path, label in test_full.samples
    ]

    # Set the class metadata to the requested order.
    train_full.classes = classes_list
    test_full.classes = classes_list

    train_full.class_to_idx = {
        name: index
        for index, name in enumerate(classes_list)
    }

    test_full.class_to_idx = {
        name: index
        for index, name in enumerate(classes_list)
    }

    # -------------------------------------------------------------------------
    # Stratified train/test split
    # -------------------------------------------------------------------------

    labels = np.array(train_full.targets)
    indices = np.arange(len(train_full))

    train_indices, test_indices = train_test_split(
        indices,
        test_size=TEST_RATIO,
        random_state=RANDOM_SEED,
        stratify=labels,
    )

    train_dataset = Subset(
        train_full,
        train_indices,
    )

    test_dataset = Subset(
        test_full,
        test_indices,
    )

    return train_dataset, test_dataset, classes_list


# =============================================================================
# Model
# =============================================================================

def create_model(num_classes: int) -> nn.Module:
    """
    Create an ImageNet-pretrained ResNet-50 classifier.

    :param num_classes: number of output classes.
    :return: ResNet-50 model.
    """
    weights = models.ResNet50_Weights.DEFAULT

    model = models.resnet50(weights=weights)

    model.fc = nn.Linear(
        model.fc.in_features,
        num_classes,
    )

    return model


# =============================================================================
# Training
# =============================================================================

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> tuple:
    """
    Train the model for one epoch.

    :param model: classification model.
    :param loader: training DataLoader.
    :param criterion: loss function.
    :param optimizer: optimizer.
    :param device: computation device.
    :return: average loss and accuracy.
    """
    model.train()

    total_loss = 0.0
    correct = 0
    total = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)

        predictions = outputs.argmax(dim=1)

        correct += (predictions == labels).sum().item()
        total += labels.size(0)

    average_loss = total_loss / total
    accuracy = correct / total

    return average_loss, accuracy


# =============================================================================
# Evaluation
# =============================================================================

def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> tuple:
    """
    Evaluate the model.

    :param model: classification model.
    :param loader: evaluation DataLoader.
    :param criterion: loss function.
    :param device: computation device.
    :return: loss, accuracy, true labels, predictions.
    """
    model.eval()

    total_loss = 0.0
    correct = 0
    total = 0

    all_labels = []
    all_predictions = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)

            predictions = outputs.argmax(dim=1)

            correct += (predictions == labels).sum().item()
            total += labels.size(0)

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predictions.cpu().numpy())

    average_loss = total_loss / total
    accuracy = correct / total

    return (
        average_loss,
        accuracy,
        np.array(all_labels),
        np.array(all_predictions),
    )


def get_device() -> torch.device:
    """
    Get the computation device (GPU, MPS, or CPU).

    :return: computation device.
    """
    if torch.cuda.is_available():
        return torch.device('cuda')
    elif torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')


def train_model(dataset_path: str, classes_list: list[str], model_output: str, num_epoch=1, imgs=512) -> None:
    """
    Train the ResNet-50 model for tree species classification.
    """
    dataset_path = Path(dataset_path)
    model_output = Path(model_output)
    set_seed(42)

    # -------------------------------------------------------------------------
    # Device
    # -------------------------------------------------------------------------

    device = get_device()
    print(f'Device: {device}')
    print()

    # -------------------------------------------------------------------------
    # Dataset
    # -------------------------------------------------------------------------

    train_dataset, test_dataset, class_names = create_datasets(dataset_path, classes_list, imgs)

    print(f'Classes: {class_names}')
    print(f'Total images: {len(train_dataset) + len(test_dataset)}')
    print(f'Training images: {len(train_dataset)}')
    print(f'Test images: {len(test_dataset)}')
    print()

    # -------------------------------------------------------------------------
    # DataLoaders
    # -------------------------------------------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )

    # -------------------------------------------------------------------------
    # Model
    # -------------------------------------------------------------------------

    model = create_model(
        num_classes=len(class_names),
    )

    model = model.to(device)

    # -------------------------------------------------------------------------
    # Loss and optimizer
    # -------------------------------------------------------------------------

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------

    best_accuracy = 0.0

    for epoch in range(num_epoch):
        train_loss, train_accuracy = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        test_loss, test_accuracy, _, _ = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device,
        )

        print(
            f'Epoch {epoch + 1:02d}/{num_epoch} | '
            f'Train Loss: {train_loss:.4f} | '
            f'Train Acc: {train_accuracy:.4f} | '
            f'Test Loss: {test_loss:.4f} | '
            f'Test Acc: {test_accuracy:.4f}'
        )

        # Save the best model according to test accuracy.
        if test_accuracy > best_accuracy:
            best_accuracy = test_accuracy
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'class_names': class_names,
                    'image_size': imgs,
                    'test_ratio': TEST_RATIO,
                    'best_test_accuracy': best_accuracy,
                },
                model_output,
            )

    # -------------------------------------------------------------------------
    # Final evaluation
    # -------------------------------------------------------------------------

    print()
    print('Loading best model...')

    checkpoint = torch.load(
        model_output,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint['model_state_dict'],
    )

    _, final_accuracy, labels, predictions = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device,
    )

    # -------------------------------------------------------------------------
    # Results
    # -------------------------------------------------------------------------

    print()
    print('=' * 80)
    print('FINAL TEST RESULTS')
    print('=' * 80)

    print(f'Test accuracy: {final_accuracy:.4f}')
    print()

    print('Classification report:')
    print(
        classification_report(
            labels,
            predictions,
            target_names=class_names,
            digits=4,
            zero_division=0,
        )
    )

    print('Confusion matrix:')
    print(
        confusion_matrix(
            labels,
            predictions,
        )
    )

    print()
    print(f'Best model saved to: {model_output}')

    return labels, predictions


def plot_confusion_matrix(labels, predictions, class_names, title='Confusion Matrix'):
    import plotly.figure_factory as ff

    cm = confusion_matrix(labels, predictions)

    plotly_cm = ff.create_annotated_heatmap(
        z=cm,
        x=class_names,
        y=class_names,
        colorscale='Blues',
        showscale=True,
        annotation_text=cm,
        hoverinfo='z',
    )

    plotly_cm.update_layout(
        title=title,
        title_x=0.5,
        xaxis_title='Predicted Label',
        yaxis_title='True Label',
        xaxis=dict(
            tickangle=-45,
            side='bottom',
        ),
        width=800,
        height=800,
    )



# -------------------------------------------------------------------------
# Prediction
# -------------------------------------------------------------------------



@dataclass
class PredictionModel:
    model: torch.nn.Module
    class_names: list
    input_size: int
    device: torch.device
    predict_transform: transforms.Compose

    def predict(self, array: np.ndarray) -> tuple:
        img = Image.fromarray(array)
        img = self.predict_transform(img)
        img = img.unsqueeze(0)
        img = img.to(self.device)
        return self.model(img)


def get_prediction_model(model_path) -> PredictionModel:
    device = get_device()
    checkpoint = torch.load(model_path, map_location=device)
    imgs = checkpoint['image_size']
    class_names = checkpoint['class_names']
    model = create_model(num_classes=len(class_names))
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()

    predict_transform = transforms.Compose([
        MirrorPadResize(imgs),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    return PredictionModel(
        model=model,
        class_names=class_names,
        input_size=imgs,
        device=device,
        predict_transform=predict_transform
    )


def predict_species(t: Tile, prediction_model: PredictionModel) -> pd.DataFrame:
    """
    predict the classes of the instances in the tile using the prediction model and return a dataframe with the results

    :param t: tile.Tile object containing the orthophoto and instance labels
    :param prediction_model: PredictionModel object containing the trained model and class names
    :return: pd.DataFrame containing the predictions for each instance in the tile
    """
    int_labels = np.unique(t.instance_labels)
    int_labels = int_labels[int_labels != 0]

    predictions = []
    for i_label in int_labels:
        mask = (t.instance_labels == i_label)
        center = nda_get_labels_centers(mask)[0]
        crop = nda_crop_using_mask(t.orthophoto, mask)

        pred = prediction_model.predict(crop)
        probabilities = torch.softmax(pred, dim=1)
        predicted_index = probabilities.argmax(dim=1).item()
        predicted_species = prediction_model.class_names[predicted_index]
        confidence = round(probabilities[0, predicted_index].item(), 3)

        predictions.append({
            'tile_instance_label': int(i_label),
            'code': predicted_species,
            'code_confidence': confidence,
            'axis-0': int(center[0]),
            'axis-1': int(center[1]),
        })

    return pd.DataFrame(predictions)


def predict_tiles_species(tile_dir: str, model_path: str, save_dir: str = 'points_pred', *, verbose: bool) -> None:
    """
    Predict the species of the instances in the tiles using the prediction model and save the results to CSV files.

    :param tile_dir: The directory containing the tiles and their orthophotos.
    :param model_path: The path to the trained prediction model.
    :param save_dir: The directory where the prediction CSV files will be saved (default is 'points_pred').
    """
    prediction_model = get_prediction_model(model_path)
    orthophoto_dir = os.path.join(tile_dir, const.ORTHOPHOTO_SUBDIR)
    tile_names = sorted(os.listdir(orthophoto_dir))

    for tile_name in tile_names:
        if not tile_name.endswith('.tif'):
            continue

        tile_name = utils_remove_extension(tile_name)
        t = tile_open_as_tile(tile_dir, tile_name)
        pred = predict_species(t, prediction_model)

        print(f'Predicting species for tile: {tile_name}') if verbose else None
        print(pred) if verbose == 2 else None
        print() if verbose else None

        save_path = os.path.join(tile_dir, save_dir)
        os.makedirs(save_path, exist_ok=True)
        pred.to_csv(os.path.join(save_path, f'{tile_name}.csv'), index=False)


