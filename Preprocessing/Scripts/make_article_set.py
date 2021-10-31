#Author: Matt Williams
#Version: 10/21/2021


from sort_dataset import sort_dataset
from web_scraper import check_if_page_exists
from constants import * 
import os
from save_load_json import load_json,save_json
from concurrent.futures import ThreadPoolExecutor
import time



def get_articles_for_category(**kwargs):
    """ A method used to return a dictionary of our chosen articles from the data set.
        The key of the dictionary is the name of a category. 
        The value of the dictionary is a list of dictionaries, where each dictionary
        contains the headline, link, and description of the article"""
    category = kwargs['category']
    delay = kwargs['delay']
    dataset = kwargs['dataset']
    article_list = []
    counter = 0

    while counter < len(dataset):
        art_obj = dataset[counter]
        art_cat = art_obj['category']
        art_link = art_obj['link']
        if art_cat == category and check_if_page_exists(art_link): 
            small_art_obj = {}
            small_art_obj['link'] = art_link
            small_art_obj['headline'] = art_obj['headline']
            small_art_obj['short_description'] = art_obj['short_description']
            article_list.append(small_art_obj)
         
        counter += 1
        print(category+ ":", len(article_list), ". Searched:", counter)
        time.sleep(delay)

    
    return category, article_list







if __name__ == "__main__": 

    if not os.path.exists(SORTED_DATA_PATH): 
        sort_dataset()

    dataset = load_json(SORTED_DATA_PATH)
    article_set = {}
    delay = 1 
    futures = []


    with ThreadPoolExecutor(max_workers= len(CATEGORIES)) as executor: 
        #start our threads
        for category in CATEGORIES:
            article_set[category] = []
            article_list = dataset[category]
            future = executor.submit(get_articles_for_category,
                                    category = category,   
                                    dataset = article_list, 
                                    delay=delay)
            futures.append(future)

    
    for future in futures:
        category, list = future.result()
        article_set[category] = list


    save_json(article_set, ARTICLE_SET_PATH)
            


    
    
    

    

