
#Author: Matt Williams
#Version: 10/14/2021
import json
import os
from constants import DATA_SET_PATH


# Method that takes in the file path of a .json file and returns the 
# python dictionary of that json file. If the path given is the original
# dataset, a list of string will be returned instead, where each string is 
# is a json object.
def load_dataset(path): 

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

    
    
    
