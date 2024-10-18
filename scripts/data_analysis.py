import json
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def load_annotations(file_path):
    """Load the JSON annotation file."""
    with open(file_path, 'r') as f:
        return json.load(f)

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

def visualize_bar(data_dict, title, xlabel, ylabel, filename):
    """Create and save bar plots."""
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

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{filename}.png")
    plt.close()

def visualize_hist(data, title, xlabel, ylabel, filename, bins=30):
    """Create and save histograms."""
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hist(data, bins=bins, color='lightblue', edgecolor='black')
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)

    # Adjust layout to prevent label cut-off
    plt.tight_layout()

    os.makedirs("plots", exist_ok=True)
    plt.savefig(f"plots/{filename}.png")
    plt.close()

def visualize_class_area_distribution(box_areas, filename_prefix):
    """Create histograms of bounding box areas by class."""
    for class_name, areas in box_areas.items():
        visualize_hist(
            areas, 
            f"Bounding Box Area Distribution for {class_name}", 
            "Area", "Frequency", 
            f"{filename_prefix}_{class_name}_area_distribution"
        )

def main(train_json_path, val_json_path):
    """Run the complete analysis on train and val datasets."""
    # Load annotations
    print("Loading Annotations", flush=True)
    train_data = load_annotations(train_json_path)
    val_data = load_annotations(val_json_path)

    # Parse data for both train and val
    train_analysis = parse_annotations(train_data)
    val_analysis = parse_annotations(val_data)

    # Visualize class distributions
    visualize_bar(train_analysis["class_counts"], 
                  "Class Distribution in Train Set", 
                  "Class", "Object Count", 
                  "train_class_distribution")
    visualize_bar(val_analysis["class_counts"], 
                  "Class Distribution in Validation Set", 
                  "Class", "Object Count", 
                  "val_class_distribution")

    # Visualize weather, scene, and time of day distributions
    visualize_bar(train_analysis["weather_counts"], 
                  "Weather Distribution in Train Set", 
                  "Weather", "Image Count", 
                  "train_weather_distribution")
    visualize_bar(val_analysis["weather_counts"], 
                  "Weather Distribution in Validation Set", 
                  "Weather", "Image Count", 
                  "val_weather_distribution")
    
    visualize_bar(train_analysis["scene_counts"], 
                  "Scene Distribution in Train Set", 
                  "Scene", "Image Count", 
                  "train_scene_distribution")
    visualize_bar(val_analysis["scene_counts"], 
                  "Scene Distribution in Validation Set", 
                  "Scene", "Image Count", 
                  "val_scene_distribution")

    visualize_bar(train_analysis["timeofday_counts"], 
                  "Time of Day Distribution in Train Set", 
                  "Time of Day", "Image Count", 
                  "train_timeofday_distribution")
    visualize_bar(val_analysis["timeofday_counts"], 
                  "Time of Day Distribution in Validation Set", 
                  "Time of Day", "Image Count", 
                  "val_timeofday_distribution")

    # Visualize bounding box area distribution for each class in train data
    visualize_class_area_distribution(train_analysis["box_areas"], "train")

    # Visualize objects per image distributions
    visualize_hist(train_analysis["boxes_per_image"], 
                   "Objects per Image in Train Set", 
                   "Number of Objects", "Frequency", 
                   "train_objects_per_image")
    visualize_hist(val_analysis["boxes_per_image"], 
                   "Objects per Image in Validation Set", 
                   "Number of Objects", "Frequency", 
                   "val_objects_per_image")

    print("All analysis completed and plots saved to the 'plots' directory.")


train_json_path = "assignment_data_bdd/bdd100k_labels_release/bdd100k_labels_images_train.json"
val_json_path = "assignment_data_bdd/bdd100k_labels_release/bdd100k_labels_images_val.json"

if __name__ == "__main__":
    main(train_json_path, val_json_path)

