# To test Custom data loader for 1 Batch of images
from custom_dataloader import create_dataloader

def test_dataloader(train_json, train_images, batch_size=4):
    """Test the custom DataLoader to ensure everything is working."""
    dataloader = create_dataloader(train_json, train_images, batch_size=batch_size)

    print("Testing DataLoader...")
    for batch_idx, (images, targets) in enumerate(dataloader):
        print(f"Batch {batch_idx + 1}")

        print(f"Image batch shape: {images.shape}")  # Should be [B, C, H, W]

        # Iterate over targets and print each target's boxes and labels
        for i, target in enumerate(targets):
            print(f"Sample {i + 1}:")
            print(f"  Boxes: {target['boxes']}")
            print(f"  Labels: {target['labels']}")

        # Exit after one batch to avoid long output
        break

if __name__ == "__main__":
    train_json = "./assignment_data_bdd/bdd100k_labels_release/bdd100k_labels_images_val.json"
    train_images = "./assignment_data_bdd/bdd100k_images_100k/val/"

    # Test the DataLoader
    test_dataloader(train_json, train_images, batch_size=4)