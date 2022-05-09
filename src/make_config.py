'''
Script to create the config file to be used by all analysis scripts.
Should be the first thing you run.
Started working: Nov. 9 2021
Author: Veronica Tarka, veronica.tarka@mail.mcgill.ca
WON'T ACCEPT ~ TO MEAN HOME
'''

# IMPORT STATEMENTS
import os.path as osp
import argparse
import json
import sys

def main():
    ###### Get the path to where 2Psinapod is installed #######
    # Option to have the path passed in when the script is run
    parser = argparse.ArgumentParser(description='Get the directory for the 2Psinapod repo')
    parser.add_argument('-d','--proj_dir', required=False, help='Path to the folder with the 2Psinapod repo')
    args = parser.parse_args()

    if args.proj_dir is not None:
        if osp.isdir(args.proj_dir):
            wd = args.proj_dir
        else:
            print("Directory provided is invalid")
            sys.exit()
    else:
        wd_input = input("Enter the path to the 2Psinapod folder (enter=default): ") or "/Users/veronica/2Psinapod"
        print(wd_input)

        if osp.isdir(wd_input):
            wd = wd_input
        else:
            print("Directory provided is invalid")
            sys.exit()


    # Check if the config file already exists, and prompt to overwrite it
    full_config_path = wd + "/config.json"
    if osp.exists(full_config_path):
        overwrite_yn = input("A config file already exists, do you want to overwrite it? ")
        overwrite_yn = overwrite_yn.lower()
        if overwrite_yn == "no" or overwrite_yn == "n":
            print("Ok, exiting script.")
            sys.exit()
        elif overwrite_yn == "yes" or overwrite_yn == "y":
            print("Ok, continuing on...")
        else:
            print("Input not understood. Expecting yes or no.")

    config_dict = {}
    config_dict['RepoDir'] = wd

    # get the name of the folder with this recording's files
    config_dict['RecordingFolder'] = input("Path to the folder with this recording\'s files: ")

    # get the name of the trigger file
    config_dict['Triggers'] = input("Name of CSV with triggers: ")

    # get the name of the conditions file
    config_dict['Conditions'] = input("Name of the CSV with condition labels: ")

    # get the name of the file to store all the analysis in
    config_dict['AnalysisFile'] = input("Name of file to store analysis in (enter=default): ") or "cells.pkl"

    # write this to the config file
    with open(full_config_path,'w') as f:
        json.dump(config_dict,f)
    

if __name__=='__main__':
    main()