#Author: Matt Williams
#Version: 11/7/2021

from constants import *
from save_load_json import load_json,save_json
import json



def sort_dataset(): 
    """The original dataset isn't sorted in any manner. This method using a dictionary to sort the articles
        in order to increase other preprocessing tasks. Each key in the dictionary is a category and each
        value is a list of article objects from the original dataset """
    dataset_lines = load_json(DATA_SET_PATH)
    sorted_dataset = {}

    counter = 0
    while counter < len(dataset_lines):
        art_obj =  json.loads(dataset_lines[counter])
        category = art_obj['category']
        if category not in sorted_dataset.keys(): 
            sorted_dataset[category] = []
        sorted_dataset[category].append(art_obj)
        counter += 1

    save_json(sorted_dataset, SORTED_DATA_PATH)




if __name__ == "__main__":
    sort_dataset()