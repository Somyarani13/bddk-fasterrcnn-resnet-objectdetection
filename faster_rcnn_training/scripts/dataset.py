import os
import json
import torch
from PIL import Image
from torch.utils.data import Dataset


def collate_fn(batch):
    """
    Custom collate function to handle batches with variable-sized targets (bounding boxes).
    """
    images, targets = list(zip(*batch))  # Unzip the batch into images and targets
    images = torch.stack(images, 0)
    return images, targets


# Custom Dataloader for Faster RCNN training
class BDDDataset(Dataset):
    """
    A custom dataset class for loading and processing the BDD (Berkeley DeepDrive) dataset.

    This dataset class reads image paths and their corresponding annotations from a JSON file.
    It processes the images and annotations to produce the necessary format for YOLO training
    and inference. It also saves YOLO-style annotations as .txt files in a specified directory.

    Args:
        json_file (str): Path to the JSON file containing the annotations.
        image_dir (str): Directory where the images are stored.
        annotations_dir (str): Directory where the YOLO-style annotation files will be saved.
        transform (callable, optional): Optional transform to be applied on an image.

    Attributes:
        data (list): A list of dictionaries containing metadata for each image in the dataset.
        image_dir (str): Path to the directory containing images.
        transform (callable): A function/transform to apply to images.
        annotations_dir (str): Path to the directory where YOLO annotations are saved.

    Methods:
        __len__: Returns the number of samples in the dataset.
        __getitem__: Fetches a sample given an index, processes the image and annotations,
                    and returns them in the required format.
        category_to_label: Maps category names to class IDs based on a custom mapping.
    """
    def __init__(self, image_dir, gt_labels_dir, flag, label_list, transforms=None):
        self.image_dir = image_dir
        self.transforms = transforms
        self.flag = flag
        
        if self.flag == 'train':
            self.img_dir = os.path.join(self.image_dir, 'train')
            self.json_dir = os.path.join(gt_labels_dir, 'bdd100k_labels_images_train.json')
        
        if self.flag == 'val':
            self.img_dir = os.path.join(self.image_dir, 'val')
            self.json_dir = os.path.join(gt_labels_dir, 'bdd100k_labels_images_val.json')
        
        print(self.img_dir, self.json_dir)
        self.names = [name[:-4] for name in
            list(filter(lambda x: x.endswith(".jpg"),
                os.listdir(self.img_dir)))]
        
        self.label_data = json.load(open(self.json_dir, 'r', encoding='UTF-8'))
        self.label_list = label_list
        
        print(f"Dataset loaded with {len(self.label_data)} for {flag}")
        print(f"Dataset loaded with {len(self.names)} for {flag}")


    def __len__(self):
        """
        Returns the number of samples in the dataset.

        Returns:
            int: The number of images/annotations in the dataset.
        """
        return len(self.names)

    def __getitem__(self, idx):
        """
        Fetches and processes a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: A tuple containing:
                - image (torch.Tensor): The processed image tensor.
                - target (dict): A dictionary containing:
                    - 'boxes' (torch.Tensor): Tensor of bounding boxes in (x_center, y_center, width, height) format.
                    - 'labels' (torch.Tensor): Tensor of labels corresponding to each bounding box.
        """
        img_name = self.names[idx]
        img_path = os.path.join(self.img_dir, img_name+ ".jpg")

        # Load the image
        image = Image.open(img_path).convert("RGB")

        # Load the image
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            print(f"Error loading image {img_name}: {e}")
            return None
        label_data = self.label_data
        boxes = []
        labels = []

        frame_labels = label_data[idx]['labels']
        for label in frame_labels:
            if "box2d" in label.keys():
                box2d = label["box2d"]
                boxes.append([box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']])
                obj_label = label['category']
                labels.append(self.label_list.index(obj_label))

        boxes = torch.tensor(boxes, dtype=torch.float32)
        labels = torch.tensor(labels, dtype=torch.long)

        target = {"boxes": boxes, "labels": labels}
        if self.transforms:
            image = self.transforms(image)
        return image, target