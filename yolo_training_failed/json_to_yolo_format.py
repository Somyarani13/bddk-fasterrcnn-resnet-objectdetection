import os
import json
import argparse
from PIL import Image

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

def convert_bbox_to_yolo(img_width, img_height, box2d):
    """
    Convert bounding box to YOLO format [x_center, y_center, width, height].
    Args:
        img_width (int): Width of the image.
        img_height (int): Height of the image.
        box2d (dict): Bounding box in the format {'x1', 'y1', 'x2', 'y2'}.
    Returns:
        list: Bounding box in YOLO format [x_center, y_center, width, height].
    """
    x1, y1 = box2d['x1'], box2d['y1']
    x2, y2 = box2d['x2'], box2d['y2']

    # Normalize to [0, 1] range
    x_center = ((x1 + x2) / 2) / img_width
    y_center = ((y1 + y2) / 2) / img_height
    width = (x2 - x1) / img_width
    height = (y2 - y1) / img_height

    return [x_center, y_center, width, height]


def convert_annotations_to_yolo(json_file, images_folder, output_folder):
    """
    Convert JSON annotations to YOLO format and save as .txt files.
    Args:
        json_file (str): Path to the JSON annotation file.
        images_folder (str): Path to the folder containing images.
        output_folder (str): Folder to save YOLO annotations.
    """
    # Ensure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Load JSON annotations
    with open(json_file, 'r') as f:
        annotations = json.load(f)

    for annotation in annotations:
        image_name = annotation['name']
        image_path = os.path.join(images_folder, image_name)

        # Skip if the image does not exist
        if not os.path.exists(image_path):
            print(f"Warning: Image not found: {image_path}. Skipping...")
            continue

        # Load image dimensions
        with open(image_path, 'rb') as img_file:
            img = Image.open(img_file)
            img_width, img_height = img.size

        yolo_annotations = []

        # Process each object in the image
        for obj in annotation.get('labels', []):
            category = obj.get('category')
            if category not in class_mapping:
                continue  # Skip unknown categories

            class_id = class_mapping[category]
            box2d = obj.get('box2d')

            if box2d:
                yolo_bbox = convert_bbox_to_yolo(img_width, img_height, box2d)
                yolo_annotations.append([class_id] + yolo_bbox)

        # Save YOLO annotations to a .txt file
        txt_filename = os.path.splitext(image_name)[0] + '.txt'
        txt_filepath = os.path.join(output_folder, txt_filename)

        with open(txt_filepath, 'w') as txt_file:
            for annotation in yolo_annotations:
                txt_file.write(" ".join(map(str, annotation)) + '\n')


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Convert BDD100K JSON annotations to YOLO format.")

    # train data
    parser.add_argument('--train_json_file', type=str, default='assignment_data_bdd/bdd100k_labels_release/bdd100k_labels_images_train.json',
                        help="Path to the Train JSON annotation file.")
    parser.add_argument('--train_images_folder', type=str, default='assignment_data_bdd/bdd100k_images_100k/train',
                        help="Path to the folder containing train images.")
    parser.add_argument('--train_output_folder', type=str, default='assignment_data_bdd/bdd100k_labels_yolo_release/train',
                    help="Folder to save Train YOLO formatted annotations.")

    # val data
    parser.add_argument('--val_json_file', type=str, default='assignment_data_bdd/bdd100k_labels_release/bdd100k_labels_images_val.json',
                        help="Path to the Train JSON annotation file.")
    parser.add_argument('--val_images_folder', type=str, default='assignment_data_bdd/bdd100k_images_100k/val',
                        help="Path to the folder containing train images.")
    parser.add_argument('--val_output_folder', type=str, default='assignment_data_bdd/bdd100k_labels_yolo_release/val',
                        help="Folder to save Val YOLO formatted annotations.")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Convert annotations to YOLO format
    print("working on train data")
    convert_annotations_to_yolo(args.train_json_file, args.train_images_folder, args.train_output_folder)

    print("working on val data")
    convert_annotations_to_yolo(args.val_json_file, args.val_images_folder, args.val_output_folder)

# python scripts/json_to_yoloformat.py
