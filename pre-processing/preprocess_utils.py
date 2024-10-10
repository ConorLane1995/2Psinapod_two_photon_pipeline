

import json

def load_config_from_json(file_path: str, config_class) -> object:
    with open(file_path, 'r') as f:
        config_data = json.load(f)
    
    # Only unpack the keys that are defined in the provided config class
    return config_class(**{key: config_data[key] for key in config_data if key in config_class.__annotations__})


