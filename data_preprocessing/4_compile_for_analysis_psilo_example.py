'''
Author:  Conor Lane, March 2024
Contact: conor.lane1995@gmail.com

PURPOSE:  Collate all of the individual recording files into four dictionaries (pre-saline, post_saline, pre-psilo, post-psilo) for easy analysis.

INPUTS:  The filepaths of the individual recordings for saline and psilo days - update as more animals are added. 
         Remember to also update the all_dicts and all_dicts_str with the names so that they can be correctly stored.

         The BASE_BATH with the filepath to the folder that will store dicts. 
'''
import time
start_time = time.time()
import numpy as np
import pickle
from dataclasses import dataclass
import os

# Import general functions from preprocess_utils file.
from preprocess_utils import load_config_from_json

# Create a class to extract the required elements from the config.json that are used here, and load as config. 
@dataclass
class Config:
    CompilationFolder: str

# Load the required variables from config.json into the config class, only if they are listed in the class.
config = load_config_from_json(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'config.json')), Config)

SAVE_PATH = config.CompilationFolder


"""
FUNCTIONS:
"""

def load_data(file_dicts):
    data = {}
    for name, file_path in file_dicts.items():
        with open(file_path, 'rb') as f:
            data[name] = pickle.load(f)
    return data


def save_dictionary(data_dict, save_path):
    
    # Get the name of the variable passed into the function
    var_name = [name for name in globals() if globals()[name] is data_dict][0]

    # Create the full file path
    file_path = os.path.join(save_path, f"{var_name}.pkl")
    
    # Save the dictionary to the pickle file
    with open(file_path, 'wb') as f:
        pickle.dump(data_dict, f)
    
    print(f"Saved {var_name} to {file_path}")


"""
FILES TO COMPILE:
"""

# SALINE PRE RECORDING

saline_pre_files = {

    "saline_pre_186": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID186/11242022_GCaMP6s_ID186_saline/TSeries-11222022-1228-021/suite2p/plane0/cells.pkl",
    "saline_pre_237": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID237/12082022_GCaMP6s_ID237_saline/TSeries-12082022-1143-031/suite2p/plane0/cells.pkl",
    "saline_pre_239": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID239/01132023_ID239_saline/TSeries-01122023-1243-038/suite2p/plane0/cells.pkl",
    "saline_pre_251": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID251/ID251_13012023_saline/TSeries-01122023-1243-040/suite2p/plane0/cells.pkl",
    "saline_pre_269": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID269/ID269_06032023_saline/TSeries-03062023-1216-061/suite2p/plane0/cells.pkl",
    "saline_pre_276": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID276/ID276_03032023_saline/TSeries-03032023-1447-059/suite2p/plane0/cells.pkl",
    "saline_pre_473": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID473/saline/TSeries-01142009-2331-135/suite2p/plane0/cells.pkl",
    "saline_pre_474": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID474/saline/TSeries-01142009-2331-137/suite2p/plane0/cells.pkl",
}

# SALINE POST RECORDING

saline_post_files = {

    "saline_post_186": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID186/11242022_GCaMP6s_ID186_saline/TSeries-11222022-1228-022/suite2p/plane0/cells.pkl",
    "saline_post_237": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID237/12082022_GCaMP6s_ID237_saline/TSeries-12082022-1143-032/suite2p/plane0/cells.pkl",
    "saline_post_239": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID239/01132023_ID239_saline/TSeries-01122023-1243-039/suite2p/plane0/cells.pkl",
    "saline_post_251": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID251/ID251_13012023_saline/TSeries-01122023-1243-041/suite2p/plane0/cells.pkl",
    "saline_post_269": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID269/ID269_06032023_saline/TSeries-03062023-1216-062/suite2p/plane0/cells.pkl",
    "saline_post_276": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID276/ID276_03032023_saline/TSeries-03032023-1447-060/suite2p/plane0/cells.pkl",
    "saline_post_473": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID473/saline/TSeries-01142009-2331-136/suite2p/plane0/cells.pkl",
    "saline_post_474": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID474/saline/TSeries-01142009-2331-138/suite2p/plane0/cells.pkl",
}



# PSILOCYBIN PRE RECORDING

psilo_pre_files = {

    "psilo_pre_186": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID186/12012022_GCaMP6s_ID186_psilo/TSeries-12012022-1321-026/suite2p/plane0/cells.pkl",
    "psilo_pre_237": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID237/12122022_GCaMP6s_ID237_psilo/TSeries-12122022-1307-033/suite2p/plane0/cells.pkl",
    "psilo_pre_239": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID239/01172023_Psilo/TSeries-01172023-1212-042/suite2p/plane0/cells.pkl",
    "psilo_pre_251": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID251/ID251_17012023_psilo/TSeries-01172023-1212-044/suite2p/plane0/cells.pkl",
    "psilo_pre_269": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID269/ID269_08032023_psilocybin/TSeries-03082023-1211-065/suite2p/plane0/cells.pkl",
    "psilo_pre_276": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID276/ID276_07032023_Psilocybin/TSeries-03072023-1303-063/suite2p/plane0/cells.pkl",
    "psilo_pre_473": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID473/psilo/TSeries-10132023-1324-139/suite2p/plane0/cells.pkl",
    "psilo_pre_474": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID474/psilo/TSeries-10132023-1324-142/suite2p/plane0/cells.pkl",

}

# PSILOCYBIN POST RECORDING

psilo_post_files = {

    "psilo_post_186": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID186/12012022_GCaMP6s_ID186_psilo/TSeries-12012022-1321-027/suite2p/plane0/cells.pkl",
    "psilo_post_237": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID237/12122022_GCaMP6s_ID237_psilo/TSeries-12122022-1307-034/suite2p/plane0/cells.pkl",
    "psilo_post_239": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID239/01172023_Psilo/TSeries-01172023-1212-043/suite2p/plane0/cells.pkl",
    "psilo_post_251": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID251/ID251_17012023_psilo/TSeries-01172023-1212-045/suite2p/plane0/cells.pkl",
    "psilo_post_269": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID269/ID269_08032023_psilocybin/TSeries-03082023-1211-066/suite2p/plane0/cells.pkl",
    "psilo_post_276": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID276/ID276_07032023_Psilocybin/TSeries-03072023-1303-064/suite2p/plane0/cells.pkl",
    "psilo_post_473": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID473/psilo/TSeries-10132023-1324-140/suite2p/plane0/cells.pkl",
    "psilo_post_474": "F:/Two-Photon/Psilocybin Project/Evoked Cohort Mice/ID474/psilo/TSeries-10132023-1324-143/suite2p/plane0/cells.pkl",
}



saline_pre_dict = load_data(saline_pre_files)
saline_post_dict = load_data(saline_post_files)

psilo_pre_dict = load_data(psilo_pre_files)
psilo_post_dict = load_data(psilo_post_files)

save_dictionary(saline_pre_dict,SAVE_PATH)
save_dictionary(saline_post_dict,SAVE_PATH)
save_dictionary(psilo_pre_dict,SAVE_PATH)
save_dictionary(psilo_post_dict,SAVE_PATH)





end_time = time.time()

# Calculate the elapsed time
elapsed_time = end_time - start_time
print(f"Runtime: {elapsed_time:.2f} seconds")