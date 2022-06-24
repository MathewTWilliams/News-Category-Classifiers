
#Author: Matt Williams
#Version: 11/7/2021
import json
import os
from utils import DATA_SET_PATH, CV_RESULTS_DIR_PATH
import re

def load_json(path): 
    """ Method that takes in the file path of a .json file and returns the 
        python dictionary of that json file. If the path given is the original
        dataset, a list of strings will be returned instead, where each string 
        is a json object. """
    if not os.path.exists(path):
        return {}


    file_path_split = path.split(".")

    #the last element in the split should be the file extention
    if not file_path_split[-1] == "json": 
        return {}

    with open(path, "r+") as file: 

        if path == DATA_SET_PATH: 
            lines = file.readlines()
            return lines
            
        else: 
            json_dict = json.load(file)
            return json_dict


def save_json(json_obj, path):
    """Given a json object and a file path, store the json object at the given file path""" 

    if path.split('.')[-1] == "json":
        if os.path.exists(path): 
            os.remove(path)


        with open(path, "w+") as file: 
            json.dump(json_obj, file, indent=1)
    
    

def load_cv_result(subfolder, word_model): 

    file_pattern = re.compile(word_model)

    path = os.path.join(CV_RESULTS_DIR_PATH, subfolder)

    for file in os.listdir(path): 
        match = file_pattern.search(file)
        if match != None: 
            return load_json(os.path.join(path, file))


    return None