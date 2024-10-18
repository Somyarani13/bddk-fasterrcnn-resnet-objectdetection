import os
import json
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision.transforms import ToTensor

class_mapping = {
    "person": 0,
    "rider": 1,
    "car": 2,
    "truck": 3,
    "bus": 4,
    "train": 5,
    "motorcycle": 6,
    "bicycle": 7,
    "traffic light": 8,
    "traffic sign": 9
}

class BDDDataset(Dataset):
    def __init__(self, json_file, images_folder, transform=None):
        """
        Custom Dataset to load BDD100K images and annotations.
        Args:
            json_file (str): Path to the JSON annotation file.
            images_folder (str): Path to the folder containing images.
            transform (callable, optional): Optional transform to apply on the image.
        """
        with open(json_file, 'r') as f:
            self.annotations = json.load(f)

        self.images_folder = images_folder
        print("Loading: ", self.images_folder, flush=True)
        self.transform = transform if transform else ToTensor()

    def __len__(self):
        """Return the total number of samples."""
        return len(self.annotations)

    def __getitem__(self, idx):
        annotation = self.annotations[idx]
        image_path = os.path.join(self.images_folder, annotation['name'])
        print("image:", image_path, annotation)

        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")

        image = Image.open(image_path).convert("RGB")
        image = self.transform(image)

        boxes, labels = [], []
        img_width, img_height = image.size(2), image.size(1)

        for obj in annotation.get('labels', []):
            category = obj.get('category')
            if category not in class_mapping:
                continue  # Skip unknown categories

            class_id = class_mapping[category]
            box2d = obj.get('box2d')
            if box2d:
                x1, y1 = box2d['x1'], box2d['y1']
                x2, y2 = box2d['x2'], box2d['y2']

                # Normalize box coordinates
                x_center = (x1 + x2) / 2 / img_width
                y_center = (y1 + y2) / 2 / img_height
                width = (x2 - x1) / img_width
                height = (y2 - y1) / img_height

                boxes.append([x_center, y_center, width, height])
                labels.append(class_id)

        if not boxes:  # Handle case where no valid boxes are found
            return self.__getitem__((idx + 1) % len(self.annotations))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.int64)

        print("annotation: ", {'image': image, 'boxes': boxes, 'labels': labels})
        return {'image': image, 'boxes': boxes, 'labels': labels}

def collate_fn(batch):
    """Custom collate function to handle batches with variable number of objects."""
    images = torch.stack([item['image'] for item in batch])  # Stack images into a tensor

    # Collect all targets as a list of dictionaries
    targets = []
    for i, item in enumerate(batch):
        target = {
            'image_id': torch.tensor([i]),  # ID of the image in the batch
            'boxes': item['boxes'],  # Tensor of shape (num_boxes, 4)
            'labels': item['labels']  # Tensor of shape (num_boxes,)
        }
        targets.append(target)

    return images, targets  # Return images and their corresponding targets

def create_dataloader(json_file, images_folder, batch_size=8, shuffle=True):
    """
    Create a DataLoader for the BDD100K dataset.
    Args:
        json_file (str): Path to the JSON annotation file.
        images_folder (str): Path to the folder containing images.
        batch_size (int, optional): Number of samples per batch. Default is 8.
        shuffle (bool, optional): Whether to shuffle the data. Default is True.
    Returns:
        DataLoader: A PyTorch DataLoader for the dataset.
    """
    dataset = BDDDataset(json_file, images_folder)
    dataloader = DataLoader(
        dataset, 
        batch_size=batch_size, 
        shuffle=shuffle, 
        collate_fn=collate_fn  # Use custom collate function
    )
    return dataloader