"""
Shared functions for use in pre_processing scripts. 
Author: Conor Lane, January 2024, conor.lane1995@gmail.com

"""

import json
import numpy as np



def load_config_from_json(file_path: str, config_class) -> object:

    """
    Load the required information from the config.json file, filtering to ensure only data requested in the script's 
    config class is imported.
    INPUT: Filepath to the json, config class.
    OUTPUT: config class containing only the required config.json parameters. 

    """

    with open(file_path, 'r') as f:
        config_data = json.load(f)
    
    # Only unpack the keys that are defined in the provided config class
    return config_class(**{key: config_data[key] for key in config_data if key in config_class.__annotations__})




def calculate_trial_avg(trial_activity):

    # Calculate the average value of the trial-response (from the 6th frame)
    return np.average(trial_activity[5:]) 

