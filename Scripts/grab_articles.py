#Author: Matt Williams
#Version: 10/13/2021


#Dataset: https://www.kaggle.com/rmisra/news-category-dataset 

from constants import * 
import json
import os

# A Method used to return a dictionary of our chosen articles from the data set.
# The Key of the dictionary is the name of a category. 
# The Value of the dictionary is a list of json objects, where each 
# json object represents an article and its respective information. 
def load_articles(): 

    data_set_map = {}
    for category in CATEGORIES:
        data_set_map[category] = []

    file_counter = 0

    search_categories = CATEGORIES.copy()

    with open(DATA_SET_NAME, "r+") as file: 
        lines = file.readlines()
        
    
        while len(search_categories) > 0 and file_counter < len(lines): 
            json_obj = json.loads(lines[file_counter])
            category = json_obj['category']

            if category in search_categories: 
                data_set_map[category].append(json_obj)

                if len(data_set_map[category]) >= ARTICLES_PER_CATEGORY: 
                    search_categories.remove(category)

            file_counter += 1


    return data_set_map

# Method used to save our chosen articles. 
def save_articles(data_set_map):

    if os.path.exists(ARTICLE_SET_NAME): 
        os.remove(ARTICLE_SET_NAME)


    with open(ARTICLE_SET_NAME, "w+") as file: 
        json.dump(data_set_map, file, indent=1)
    


if __name__ == "__main__": 
    data_set_map = load_articles()
    save_articles(data_set_map)
   


    
    
    

    

