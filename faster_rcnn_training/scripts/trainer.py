"""
Module for training and evaluating a model on the BDD100K dataset.
"""
import os
import sys
sys.path.append('.')
from scripts.dataset import BDDDataset

import torch
import torchvision
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

# Custom BDD Trainer class
class BDD_Trainer:
    def __init__(self, model, optimizer, lr_scheduler, train_loader, val_loader, device):
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.device = device
        self.model.to(self.device)

    def train(self):
        self.model.train()
        print(f'Starting training...')
        total_loss = []
        total_batches = len(self.train_loader)

        for batch_idx, (images, targets) in enumerate(self.train_loader):
            images = list(image.to(self.device) for image in images)
            targets = [{k: v.to(self.device) for k, v in t.items()} for t in targets]

            print(f'Performing forward pass for batch {batch_idx + 1}/{total_batches}')
            loss_dict = self.model(images, targets)

            losses = sum(loss for loss in loss_dict.values())
            self.optimizer.zero_grad()
            losses.backward()
            self.optimizer.step()
            self.lr_scheduler.step()
            total_loss.append(losses.cpu().detach().numpy())
            loss=losses.item()

            if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == total_batches:
                print(f"Epoch Progress: [{batch_idx + 1}/{total_batches}] - Loss: {loss:.4f}")

        train_loss = sum(total_loss) / len(total_loss)
        print(f'Epoch completed with average loss: {train_loss:.4f}')
        return train_loss
    

def get_model(model_name, num_classes):
    """
    Returns an object detection model based on the model_name.

    Parameters:
    - model_name (str): The name of the model architecture.
    - num_classes (int): The number of classes
    Returns:
    - A PyTorch model ready for training.
    """
    # Load a pre-trained Faster R-CNN model for fine-tuning
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(
        weights="DEFAULT",
        pretrained=True,
        max_detections_per_img=50)

    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model


def train(config, bdd_objects, model_save_path):
    """
    Train and evaluate a model.

    Parameters:
    - config (dict): Dictionary containing configuration parameters for training.
    - model_save_path (str): Path to save trained model weights.

    Returns:
    - None
    """
    lr = config.config['lr']
    batch_size = config.config['batch_size']
    epochs = config.config['epochs']
    base_dir = config.config['img_dir']
    labels_dir = config.config['label_dir']
    model_name = config.config['model_name']
    train_model_flag =  config.config['train']
    train_images =  config.config['train_images']
    device =config.config['device']
    val_images =  config.config['val_images']

    train_transform = T.Compose([T.ToTensor()])
    num_classes = len(bdd_objects)

    if not os.path.exists(model_save_path):
        os.makedirs(model_save_path)

    train_set = BDDDataset(image_dir=base_dir,
                        gt_labels_dir=labels_dir,
                        transforms=train_transform,
                        flag='train',
                        label_list=bdd_objects)
    val_set = BDDDataset(image_dir=base_dir,
                        gt_labels_dir=labels_dir,
                        transforms=train_transform,
                        flag='val',
                        label_list=bdd_objects)

    train_loader = DataLoader(train_set,
                        batch_size=batch_size,
                        collate_fn=lambda x: tuple(zip(*x)),
                        shuffle=True)
    val_loader = DataLoader(val_set,
                        batch_size=batch_size,
                        collate_fn=lambda x: tuple(zip(*x)))

    model = get_model(model_name, num_classes)
    model.to(device)

    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=0.0005)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    trainer = BDD_Trainer(model, optimizer, lr_scheduler, train_loader, val_loader, device)

    if train_model_flag:
        # train and validation
        for epoch in range(epochs):
            t_loss = trainer.train()
            print(f"Epoch {epoch+1}/{epochs}, Train Loss: {t_loss:.4f}")
        torch.save(model.state_dict(), os.path.join(model_save_path, f'fasterrcnn_model.pth'))

        print("Training and evaluation completed.")