#Author: Matt Williams
#Version: 10/15/2021


from sort_dataset import sort_dataset
from web_scraper import check_if_page_exists
from constants import * 
import json
import os
from save_load_json import load_json,save_json
import threading
import time

#think about storting dataset
class ArticleSetThread(threading.Thread):

    def __init__(self, category, dataset, delay):
        threading.Thread.__init__(self)
        self.article_list = []
        self.category = category
        self.dataset = dataset
        self.delay = delay

    def run(self): 
        counter = 0
        while len(self.article_list) < ARTICLES_PER_CATEGORY and counter < len(self.dataset):
            art_obj = self.dataset[counter]
            art_cat = art_obj['category'] 
            art_link = art_obj['link']
            if art_cat == self.category and check_if_page_exists(art_link): 
                small_art_obj = {}
                small_art_obj['link'] = art_link
                small_art_obj['headline'] = art_obj['headline']
                small_art_obj['short_description'] = art_obj['short_description']
                self.article_list.append(small_art_obj)
            counter += 1
            print(self.category + ":", len(self.article_list))
            time.sleep(self.delay)

    def get_article_list(self):
        return self.article_list

    def get_category(self):
        return self.category




# A Method used to return a dictionary of our chosen articles from the data set.
# The Key of the dictionary is the name of a category. 
# The Value of the dictionary is a list of dictionaries, where each dictionary
# contains the headline, link, and description of the article

if __name__ == "__main__": 

    if not os.path.exists(SORTED_DATA_PATH): 
        sort_dataset()

    dataset = load_json(SORTED_DATA_PATH)
    threads = []
    article_set = {}
    delay = 0.25

    #start our threads
    for category in CATEGORIES:
        article_set[category] = []
        thread = ArticleSetThread(category, dataset[category], delay)
        threads.append(thread)
        thread.start()


    #wait for all threads to finish
    for thread in threads: 
        thread.join()

    for thread in threads: 
        article_set[thread.get_category()] = thread.get_article_list()


    save_json(article_set, ARTICLE_SET_PATH)
            


    
    
    

    

