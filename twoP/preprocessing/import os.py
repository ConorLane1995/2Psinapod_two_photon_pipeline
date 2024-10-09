import os
import sys
import json

try:
    # Determine the current directory and config path
    current_dir = os.path.abspath(os.path.dirname(__file__))
    config_path = os.path.join(current_dir, '../../../config.json')

    # Load the configuration
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Print the configuration for inspection
    print("Configuration:")
    print(json.dumps(config, indent=4))

    # Check and print the current working directory
    print("Current directory:", os.getcwd())

    # Check and print the sys.path
    print("sys.path:", sys.path)

    # Check and print PYTHONPATH environment variable
    print("PYTHONPATH:", os.environ.get('PYTHONPATH'))

    # Append the necessary path to sys.path
    parent_dir = os.path.join(current_dir, '..')
    sys.path.append(parent_dir)
    print(f"Appended parent directory: {parent_dir}")

    # Attempt to import the module
    from utils import get_active_cells
    print("Module imported successfully.")
except Exception as e:
    print(f"An error occurred: {e}")