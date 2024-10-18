import argparse
import torch
from ultralytics import YOLO
from custom_dataloader import create_dataloader

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Train YOLOv8 on BDD100K Dataset using Custom DataLoader")

    # Dataset paths
    parser.add_argument('--train_json', type=str, required=True,
                        help="Path to the training JSON annotation file.")
    parser.add_argument('--train_images', type=str, required=True,
                        help="Path to the training images folder.")
    parser.add_argument('--val_json', type=str, required=True,
                        help="Path to the validation JSON annotation file.")
    parser.add_argument('--val_images', type=str, required=True,
                        help="Path to the validation images folder.")

    # Training hyperparameters
    parser.add_argument('--epochs', type=int, default=5, help="Number of epochs to train the model.")
    parser.add_argument('--batch_size', type=int, default=8, help="Batch size for training.")
    parser.add_argument('--img_size', type=int, default=640, help="Image size for training.")
    parser.add_argument('--learning_rate', type=float, default=0.001, help="Learning rate for the optimizer.")

    # Model path and save path
    parser.add_argument('--model', type=str, default='yolov8n.pt',
                        help="Pre-trained YOLOv8 model to fine-tune.")
    parser.add_argument('--save_model', type=str, default='yolov8_bdd100k.pt',
                        help="Path to save the trained model.")
    parser.add_argument('--data', type=str, default='bdd100k.yaml', help='Path to the dataset YAML configuration file.')

    return parser.parse_args()


def format_targets_for_yolo(targets):
    """Format targets into the structure expected by YOLOv8."""
    formatted_targets = []
    for i, target in enumerate(targets):
        boxes = target['boxes']
        labels = target['labels'].unsqueeze(1)  # Reshape labels to [N, 1]
        batch_index = torch.full((boxes.size(0), 1), i, dtype=torch.float32)  # [N, 1]

        # Concatenate batch index, labels, and boxes into the required format
        formatted_target = torch.cat((batch_index, labels, boxes), dim=1)
        formatted_targets.append(formatted_target)

    # Combine all targets into a single tensor
    return torch.cat(formatted_targets, dim=0)

def train_one_epoch(model, dataloader, optimizer, device, args):
    """Train the model for one epoch."""
    model.train(data=args.data, epochs=args.epochs, batch=args.batch_size, imgsz=args.img_size)
    total_loss = 0

    for images, targets in dataloader:
        images = images.to(device)
        formatted_targets = format_targets_for_yolo(targets).to(device)

        optimizer.zero_grad()
        results = model(images, targets=formatted_targets)  # Directly pass targets
        loss = results[0]  # Extract the loss from the results

        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Training Loss: {avg_loss:.4f}")


def validate(model, dataloader, device):
    """Validate the model on the validation set."""
    model.eval()
    total_loss = 0

    with torch.no_grad():
        for images, targets in dataloader:
            images = images.to(device)
            formatted_targets = format_targets_for_yolo(targets).to(device)

            outputs = model(images)
            loss = model.loss(outputs, formatted_targets)

            total_loss += loss.item()

    avg_loss = total_loss / len(dataloader)
    print(f"Validation Loss: {avg_loss:.4f}")

def main():
    args = parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the YOLOv8 model
    print(f"Loading model: {args.model}")
    model = YOLO(args.model).to(device)

    # Create DataLoaders for training and validation
    print("Creating DataLoaders...")
    train_loader = create_dataloader(args.train_json, args.train_images, args.batch_size)
    val_loader = create_dataloader(args.val_json, args.val_images, args.batch_size, shuffle=False)

    # Set up the optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    # Custom training loop
    for epoch in range(args.epochs):
        print(f"Epoch {epoch + 1}/{args.epochs}")
        train_one_epoch(model, train_loader, optimizer, device, args)
        validate(model, val_loader, device)

    model.save(args.save_model)
    print(f"Training completed. Model saved as '{args.save_model}'.")


if __name__ == "__main__":
    main()


