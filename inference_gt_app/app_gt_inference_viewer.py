import streamlit as st
import os
import json
import numpy as np
from PIL import Image
import cv2
from mmdet.apis import init_detector, inference_detector

# Set up paths to model config and checkpoint files
config_file = './inference_gt_app/configs/fast_rcnn_bdd100/faster_rcnn_r50_fpn_1x_det_bdd100k.py'
checkpoint_file = './inference_gt_app/configs/fast_rcnn_bdd100/faster_rcnn_r50_fpn_1x_det_bdd100k.pth'

# Initialize the Faster RCNN model trained on BDD100k
model = init_detector(config_file, checkpoint_file, device='cpu')  # Use 'cuda:0' for GPU

# Load validation annotations (val.json) containing ground truth (GT) data
val_json_path = "./assignment_data_bdd/bdd100k_labels_release/bdd100k_labels_images_val.json"
with open(val_json_path, 'r') as f:
    val_annotations = json.load(f)

def draw_gt_boxes(image, gt_labels, color=(0, 255, 0)):
    """
    Draws the ground truth bounding boxes with class labels on the given image.

    Parameters:
        image (np.ndarray): The input image as a NumPy array.
        gt_labels (list): A list of dictionaries containing bounding box coordinates and class labels.
        color (tuple): Color of the bounding box and text (default is green).
    
    Returns:
        np.ndarray: The image with ground truth boxes and labels drawn on it.
    """
    for gt_label in gt_labels:
        box = gt_label['box']  # Extract bounding box coordinates
        class_label = gt_label['category']  # Extract the class label
        x1, y1, x2, y2 = map(int, box)  # Convert coordinates to integers
        # Draw the bounding box
        cv2.rectangle(image, (x1, y1), (x2, y2), color, 1)
        # Draw the class label
        cv2.putText(image, class_label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 1)
    return image

def get_gt_boxes_and_labels_for_image(image_name, annotations):
    """
    Extracts ground truth bounding boxes and class labels for a specific image from the annotations.

    Parameters:
        image_name (str): The name of the image to extract GT labels for.
        annotations (list): A list of annotation dictionaries loaded from val.json.

    Returns:
        list: A list of ground truth labels, each containing a 'box' (bounding box coordinates) and 'category' (class label).
    """
    for item in annotations:
        if item['name'] == image_name:  # Match the image by its name in annotations
            gt_labels = []
            for label in item['labels']:
                if 'box2d' in label:  # If the label contains a 2D bounding box
                    box2d = label['box2d']
                    box = [box2d['x1'], box2d['y1'], box2d['x2'], box2d['y2']]
                    category = label['category']  # Extract the class label
                    gt_labels.append({'box': box, 'category': category})  # Store box and category
            return gt_labels  # Return the list of GT labels
    return []  # Return empty list if no GT found for the image

# Streamlit web app for BDD100K model inference and GT visualization
st.title("BDD100K Inference and GT Viewer")
st.subheader("Faster RCNN trained on the BDD100K Dataset")
st.divider()
st.write("**NOTE**: Upload an image to perform inference. If the image is from the validation set, ground truth annotations will also be shown.")

uploaded_file = st.file_uploader("Choose an image...", type="jpg")

# Paths for validation images and output storage
val_images_path = "./assignment_data_bdd/bdd100k_images_100k/val"
output_images_folder = "./inference_gt_app/gt_inference_results"

if uploaded_file is not None:
    # Convert the uploaded file into an image format that can be processed
    image = Image.open(uploaded_file)
    img_np = np.array(image)  # Convert to NumPy array

    # Display the uploaded image
    st.image(img_np, caption="Uploaded Image", use_column_width=True)
    st.write("Inferencing...")

    # Perform model inference on the uploaded image
    result = inference_detector(model, img_np)

    # Check if the image is part of the validation set for GT annotations
    file_name = uploaded_file.name
    got_gt = False
    gt_img = img_np.copy()  # Copy the image for GT drawing

    if file_name in os.listdir(val_images_path):  # If the image is in the validation set
        got_gt = True
        gt_labels = get_gt_boxes_and_labels_for_image(file_name, val_annotations)
        gt_img = draw_gt_boxes(gt_img, gt_labels, color=(0, 255, 0))

    # Draw inference results (bounding boxes from the model)
    result_img = model.show_result(img_np, result)

    # Display inference results and ground truth annotations
    st.image(result_img, caption="Inference Results", use_column_width=True)
    
    if got_gt:
        st.write("Ground truth annotations found. Displaying...")
    else:
        st.write("No ground truth annotations found for the input image.")
    
    st.image(gt_img, caption="GT Annotations", use_column_width=True)

    # Save the result images (both inference and GT)
    os.makedirs(output_images_folder, exist_ok=True)  # Create output directory if not present
    inference_output_path = os.path.join(output_images_folder, f"inference_{uploaded_file.name}")
    gt_output_path = os.path.join(output_images_folder, f"gt_{uploaded_file.name}")

    # Save inference and GTresult image
    cv2.imwrite(inference_output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
    cv2.imwrite(gt_output_path, cv2.cvtColor(gt_img, cv2.COLOR_RGB2BGR))

    # Inform the user about the saved files
    st.write(f"**Inference Result** saved at: {inference_output_path}")
    st.write(f"**GT Annotations** saved at: {gt_output_path}")
