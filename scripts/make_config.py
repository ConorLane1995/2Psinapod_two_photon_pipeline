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
import sys

def main():
    ###### Get the path to where 2Psinapod is installed #######
    # Option to have the path passed in when the script is run
    parser = argparse.ArgumentParser(description='Get the directory for the 2P analysis scripts')
    parser.add_argument('-d','--proj_dir', required=False, help='Path to the 2Psinapod folder')
    args = parser.parse_args()

    if args.proj_dir is not None:
        if osp.isdir(args.proj_dir):
            wd = args.proj_dir
        else:
            print("Directory provided is invalid")
            sys.exit()
    else:
        wd_input = input("Enter the path to the 2Psinapod folder: ")

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
        if overwrite_yn == "no":
            print("Ok, exiting script.")
            sys.exit()
        elif overwrite_yn == "yes":
            print("Ok, continuing on...")
        else:
            print("Input not understood. Expecting yes or no.")

    config_dict = {}
    config_dict['WorkingDir'] = wd


if __name__=='__main__':
    main()