from constants import *
from save_load_json import load_json,save_json
import json



def sort_dataset(): 
    """ Take our original dataset and sort the articles using a dictionary.
        Then save that to a file. Done to improve preprocessing speed. """
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