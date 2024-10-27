# List of target objects for detection
bdd_objects = [
    'bus', 'traffic light', 'traffic sign',
    'person', 'bike', 'truck', 'motor', 'car', 'train', 'rider'
]
 
config = dict(
    lr=0.0001,      # Learning rate.
    batch_size=2,   # Number of images per batch.
    epochs=1,       # Number of training epochs.
    model_name="fasterrcnn_resnet50_fpn",                       # Model architecture.
    img_dir="assignment_data_bdd/bdd100k_images_100k",       # Path to the images.
    label_dir="assignment_data_bdd/bdd100k_labels_release",  # Path to the labels.
    train=True,            # Flag to indicate training mode.
    prob_threshold=0.02,   # Min confidence for detection.
    iou_threshold=0.5,     # IoU threshold for non-max suppression.
    train_images=70000,    # Number of images for training.
    val_images=10000,      # Number of images for validation.
    device='cpu'          # Device to run on: 'cuda' (GPU) or 'cpu'.
)
