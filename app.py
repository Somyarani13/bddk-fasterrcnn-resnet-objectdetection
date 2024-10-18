import streamlit as st
import json
import cv2
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_annotations(file_path):
    """Load the JSON annotation file from disk."""
    with open(file_path, 'r') as f:
        return json.load(f)

def draw_annotations(image_path, labels):
    """Draw bounding boxes and labels on the image."""
    image = cv2.imread(image_path)
    if image is None:
        st.warning(f"Image not found: {image_path}")
        return None

    for label in labels:
        category = label.get("category", "unknown")
        box = label.get("box2d", None)
        if box:
            x1, y1, x2, y2 = int(box["x1"]), int(box["y1"]), int(box["x2"]), int(box["y2"])
            # Draw rectangle
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            # Put class label
            cv2.putText(image, category, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                        0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Convert BGR image to RGB for Streamlit display
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

def visualize_unique_samples(data, image_folder, dataset_name):
    """Display 10 unique annotated samples from the dataset."""
    st.header(f"Samples from {dataset_name} Set")

    # Randomly sample 10 images from the dataset
    sampled_images = np.random.choice(data, 10, replace=False)

    for image_data in sampled_images:
        image_name = image_data["name"]
        image_path = os.path.join(image_folder, image_name)

        # Draw annotations and display the image
        annotated_image = draw_annotations(image_path, image_data.get("labels", []))
        if annotated_image is not None:
            st.image(annotated_image, caption=image_name, use_column_width=True)

def parse_annotations(data):
    """Extract object detection classes and their attributes."""
    class_counts = defaultdict(int)
    box_areas = defaultdict(list)  # Store areas by class
    weather_counts = defaultdict(int)
    scene_counts = defaultdict(int)
    timeofday_counts = defaultdict(int)
    boxes_per_image = []

    for image_data in data:
        labels = image_data.get('labels', [])
        boxes_per_image.append(len(labels))  # Number of objects per image

        # Extract weather, scene, and time of day attributes
        weather = image_data["attributes"].get("weather", "unknown")
        scene = image_data["attributes"].get("scene", "unknown")
        timeofday = image_data["attributes"].get("timeofday", "unknown")
        weather_counts[weather] += 1
        scene_counts[scene] += 1
        timeofday_counts[timeofday] += 1

        # Process bounding boxes and classes
        for label in labels:
            category = label.get('category', None)
            if category:
                class_counts[category] += 1

            # Calculate area of bounding boxes by class
            box = label.get('box2d', {})
            if box:
                width = box["x2"] - box["x1"]
                height = box["y2"] - box["y1"]
                area = width * height
                box_areas[category].append(area)

    return {
        "class_counts": class_counts,
        "box_areas": box_areas,
        "weather_counts": weather_counts,
        "scene_counts": scene_counts,
        "timeofday_counts": timeofday_counts,
        "boxes_per_image": boxes_per_image
    }

def add_percentage_labels(ax):
    """Add percentage labels on top of the bars in a bar chart."""
    total = sum([p.get_height() for p in ax.patches if p.get_height() > 0])
    for p in ax.patches:
        height = p.get_height()
        if height > 0:
            percentage = f"{(height / total) * 100:.2f}%"
            ax.annotate(percentage, 
                        (p.get_x() + p.get_width() / 2., height),
                        ha='center', va='center', 
                        xytext=(0, 9), textcoords='offset points')

def visualize_bar(data_dict, title, xlabel, ylabel):
    """Create and display bar plots in Streamlit."""
    df = pd.DataFrame(list(data_dict.items()), columns=[xlabel, ylabel])
    df = df.sort_values(by=ylabel, ascending=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    ax.bar(df[xlabel], df[ylabel], color='lightblue', edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.xticks(rotation=45, ha='right')

    add_percentage_labels(ax)  # Add percentage labels to the bars

    # Adjust layout to prevent label cut-off
    plt.tight_layout()
    st.pyplot(fig)

def visualize_hist(data, title, xlabel, ylabel, bins=30):
    """Create and display histograms in Streamlit."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data, bins=bins, color='lightblue', edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Adjust layout to prevent label cut-off
    plt.tight_layout()
    st.pyplot(fig)

def visualize_class_area_distribution(box_areas):
    """Create histograms of bounding box areas by class in Streamlit."""
    for class_name, areas in box_areas.items():
        st.subheader(f"Bounding Box Area Distribution for {class_name} - Train Set")
        visualize_hist(
            areas, 
            f"Bounding Box Area Distribution for {class_name}", 
            "Area", "Frequency", 
            bins=30
        )

def display_conclusions():
    """Display key insights and conclusions based on the dataset analysis."""
    st.header("Key Insights and Conclusions")

    st.write("""
    ### 1. Class Imbalance and Weight Adjustment
    - Some object classes (like cars or pedestrians) may dominate the dataset, while others (like bikes or buses) might have very few instances.
    - **Impact**: Models may perform poorly on underrepresented classes.
    - **Solution**:
        - Use **class weights** to give more importance to rare classes during training.
        - Alternatively, **oversample underrepresented classes** or **use synthetic data augmentation** for balancing.

    ### 2. Anchor Box Sizes and Model Architectures
    - Bounding box area distributions suggest that most objects are **small** (e.g., distant signs or pedestrians).
    - **Impact**: Default anchor box sizes in models like YOLO or Faster R-CNN might not capture small objects effectively.
    - **Solution**:
        - Adjust **anchor box sizes** to match the size of objects in the dataset.
        - Use models optimized for small objects (e.g., **YOLOv5s** or **EfficientDet**).

    ### 3. Overrepresented Weather and Scene Conditions
    - The dataset is dominated by **daytime and clear weather scenes**, while other conditions like **night or rainy scenes** are underrepresented.
    - **Impact**: The model might not generalize well to unseen conditions (e.g., foggy weather).
    - **Solution**:
        - Use **data augmentation** techniques such as:
            - Brightness/contrast adjustments to simulate night.
            - Gaussian noise to simulate fog.
            - Motion blur to simulate rain.
        - Consider using **domain adaptation techniques** for handling weather variability.

    ### 4. Objects Per Image and Model Efficiency
    - Some images contain **many objects** (e.g., crowded traffic), which can increase the complexity of detection.
    - **Impact**: 
        - Crowded scenes could lead to overlapping bounding boxes, resulting in **label noise** and poor model performance.
        - Models may also take longer to train and infer due to more objects per image.
    - **Solution**:
        - Use **Soft-NMS** or **Weighted Box Fusion (WBF)** to reduce the impact of overlapping boxes.
        - Consider models with **multi-scale prediction heads** (e.g., RetinaNet or EfficientDet).

    ### 5. Train vs. Validation Distribution Imbalance
    - If the distributions between train and validation sets differ significantly (e.g., more buses in train but fewer in val), the model may overfit.
    - **Solution**:
        - Ensure **similar distributions** across train and validation datasets.
        - Monitor performance on both train and validation sets using **per-class metrics**.

    ### 6. Small Object Detection Challenges
    - Small objects like **traffic signs** are harder to detect, especially at a distance.
    - **Impact**: Detection models might perform poorly on small objects without proper tuning.
    - **Solution**:
        - Increase the **input image resolution** during training (e.g., 640x640 instead of 416x416).
        - Use **augmentation techniques** like zoom-in cropping to focus on small objects.

    ### 7. Batch Size and Learning Rate Adjustment
    - The presence of many small objects and dense scenes might lead to noisy gradients during training.
    - **Impact**: Large batch sizes might make training unstable due to gradient noise.
    - **Solution**:
        - Use **smaller batch sizes** with **gradient accumulation** to stabilize training.
        - Lower the **initial learning rate** and use a **cyclical learning rate scheduler** to help the model converge smoothly.

    ### 8. Choice of Loss Functions and Metrics
    - Class imbalance and small object detection require careful choice of loss functions.
    - **Solution**:
        - Use **Focal Loss** to handle class imbalance by down-weighting easy examples.
        - Consider using **IoU-based losses** (like **GIoU** or **DIoU**) to improve the accuracy of bounding box predictions.

    ### 9. Evaluation Metrics for Model Performance
    - The dataset contains both **frequent and rare objects**. A simple mAP score might not reveal performance issues for underrepresented classes.
    - **Solution**:
        - Use **mAP per class** to evaluate model performance across all classes.
        - Monitor **precision and recall** metrics to ensure the model is not biased towards frequent classes.
    """)

def run_analysis(file_path, dataset_name):
    """Run analysis and display visualizations for the given dataset."""
    data = load_annotations(file_path)
    analysis = parse_annotations(data)

    st.subheader(f"Class Distribution - {dataset_name} Set")
    visualize_bar(analysis["class_counts"], 
                  f"Class Distribution in {dataset_name} Set", 
                  "Class", "Object Count")

    st.subheader(f"Weather Distribution - {dataset_name} Set")
    visualize_bar(analysis["weather_counts"], 
                  f"Weather Distribution in {dataset_name} Set", 
                  "Weather", "Image Count")

    st.subheader(f"Scene Distribution - {dataset_name} Set")
    visualize_bar(analysis["scene_counts"], 
                  f"Scene Distribution in {dataset_name} Set", 
                  "Scene", "Image Count")

    st.subheader(f"Time of Day Distribution - {dataset_name} Set")
    visualize_bar(analysis["timeofday_counts"], 
                  f"Time of Day Distribution in {dataset_name} Set", 
                  "Time of Day", "Image Count")

    st.subheader(f"Objects per Image - {dataset_name} Set")
    visualize_hist(analysis["boxes_per_image"], 
                   f"Objects per Image in {dataset_name} Set", 
                   "Number of Objects", "Frequency")

def main():
    st.title("BDD100K Dataset Analysis")

    # Input paths to the JSON files
    train_file_path = st.text_input("Enter Train JSON File Path", "")
    val_file_path = st.text_input("Enter Validation JSON File Path", "")
    image_folder = st.text_input("Enter Train Image Folder Path", "")

    if train_file_path:
        st.header("Train Dataset Analysis")
        run_analysis(train_file_path, "Train")

    if val_file_path:
        st.header("Validation Dataset Analysis")
        run_analysis(val_file_path, "Validation")

        # Display key insights and conclusions
        display_conclusions()

    # Per-class analysis for train data (only if train data path is provided)
    if train_file_path:
        st.header("Per-Class Analysis - Train Set")
        data = load_annotations(train_file_path)
        analysis = parse_annotations(data)
        visualize_class_area_distribution(analysis["box_areas"])

    # Visualize unique samples from the train set
    if image_folder:
        visualize_unique_samples(data, image_folder, "Train")

if __name__ == "__main__":
    main()
