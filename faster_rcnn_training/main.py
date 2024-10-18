import sys
sys.path.append('.')

import os
import argparse
from scripts.trainer import train, get_model
from importlib.machinery import SourceFileLoader

def parse_arguments():
    parser = argparse.ArgumentParser()
    
    parser.add_argument("--path_to_model_config", required=True, type=str)        # Path to the config file.
    parser.add_argument("--path_to_save_trained_model", required=True, type=str)  # Path to save the trained model.
    parser.add_argument("--path_to_train_dir", required=True, type=str)             # Directory of validation images.
    parser.add_argument("--path_to_train_json", required=True, type=str)            # Path to validation labels in JSON format.
    parser.add_argument("--device", required=True, type=str)                      # Device for computation ('cuda' or 'cpu').
    return parser.parse_args()

def main(config, bdd_objects, save_model_path):
    train(config, bdd_objects, save_model_path)

if __name__ == '__main__':
    args = parse_arguments()  
    print("Model Config:", args.path_to_model_config)

    # Assign parsed arguments to variables.
    config_path = args.path_to_model_config
    path_to_save_trained_model = args.path_to_save_trained_model
    path_to_train_dir = args.path_to_train_dir
    path_to_train_json = args.path_to_train_json
    output_folder = args.output_folder
    device = args.device

    model_name = 'fasterrcnn_resnet50_fpn'                    # Hardcoded model name.
    num_classes = 10                                          # Number of classes
    model_class = lambda: get_model(model_name, num_classes)  # Create a callable to get the model instance.

    # Check if the config file exists.
    if not os.path.exists(config_path):
        raise RuntimeError(f"Config '{config_path}' does not exist")

    # Load the config dynamically using SourceFileLoader.
    config = SourceFileLoader(args.path_to_model_config, config_path).load_module()

    # Access the list of target objects and the run type from the config.
    bdd_objects = config.bdd_objects
    main(config, bdd_objects, path_to_save_trained_model)
