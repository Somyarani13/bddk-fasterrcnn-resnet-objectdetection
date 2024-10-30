import cv2
import json
import os

def load_annotations(json_file_path, image_name):
    """Load annotations for a specific image from the JSON file."""
    with open(json_file_path, 'r') as f:
        data = json.load(f)

    # Find the annotations for the specific image
    for image_data in data:
        if image_data["name"] == image_name:
            return image_data["labels"]
    return None


def draw_annotations_on_image(image_path, json_file_path, output_folder):
    """Draw bounding boxes and class names on the image."""
    image_name = os.path.basename(image_path)
    labels = load_annotations(json_file_path, image_name)

    if labels is None:
        print(f"No annotations found for image: {image_name}")
        return

    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print(f"Image not found: {image_path}")
        return

    # Draw each label's bounding box and class name
    for label in labels:
        category = label.get("category", "unknown")
        box = label.get("box2d", None)

        if box:
            x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
            # Draw the bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put the class label text
            cv2.putText(image, category, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Save the output image with the same filename in the output folder
    output_path = os.path.join(output_folder, image_name)
    cv2.imwrite(output_path, image)
    print(f"Annotated image saved at: {output_path}")

# Input, Output image and annotations Path
input_folder = "./assignment_data_bdd/bdd100k_images_100k/val"
output_folder = "./issue_images"
json_path = "./assignment_data_bdd/bdd100k_labels_release/bdd100k_labels_images_val.json"

input_image_name = "7d128593-0ccfea4c.jpg"

draw_annotations_on_image(os.path.join(input_folder, input_image_name), json_path, output_folder)

# Names of images with issues in val folder
# "c9ca4c6c-4c9db372.jpg"