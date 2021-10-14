#Author: Matt Williams
#Version: 10/15/2021


from constants import * 
import json
import os
from load_dataset import load_dataset

# A Method used to return a dictionary of our chosen articles from the data set.
# The Key of the dictionary is the name of a category. 
# The Value of the dictionary is a list of strings, where each string
# is a link to an article.  
def make_article_set(): 

    article_set = {}
    for category in CATEGORIES:
        article_set[category] = []

    file_counter = 0

    search_categories = CATEGORIES.copy()

    lines = load_dataset(DATA_SET_PATH)
        
    
    while len(search_categories) > 0 and file_counter < len(lines): 
        art_obj = json.loads(lines[file_counter])
        category = art_obj['category']

        if category in search_categories: 
            article_set[category].append(art_obj['link'])

            if len(article_set[category]) >= ARTICLES_PER_CATEGORY: 
                search_categories.remove(category)

        file_counter += 1


    return article_set

# Method used to save our chosen articles. 
def save_json_obj(json_obj, path):

    if os.path.exists(path): 
        os.remove(path)


    with open(path, "w+") as file: 
        json.dump(json_obj, file, indent=1)
    



if __name__ == "__main__": 
    article_set = make_article_set()
    save_json_obj(article_set, ARTICLE_SET_PATH)
   


    
    
    

    

