#Author: Matt Williams
#Version: 11/7/2021


from web_scraper import check_if_page_exists
from constants import * 
from save_load_json import load_json,save_json
from concurrent.futures import ThreadPoolExecutor
import time



def get_articles_for_category(**kwargs):
    """ Given a list of keyword arguments which should contain a category, time delay, and 
        a list of articles for the given category, return the given category and a sub set of the list
        of articles. The dataset returned is a list of json objects where each object contains the
        link, headline, and description of the article. Each article link is checked via a head
        request to make sure the article is still accessible. If it is not, that article
        is not added to the final article set to be returned."""
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




def make_article_set():
    """The main method for this script. Load in our Sorted Data set. Then using
        multi-threading and the method above to make our article sets for each category.
        Then save the article to a json file.  """ 

    dataset = load_json(SORTED_DATA_PATH)
    article_set = {}
    delay = 1 
    futures = []


    with ThreadPoolExecutor(max_workers= len(CATEGORIES)) as executor: 
        
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



if __name__ == "__main__": 
    make_article_set()
            


    
    
    

    

