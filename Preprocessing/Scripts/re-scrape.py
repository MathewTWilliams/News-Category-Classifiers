# While scrapping the selected articles, on further inspection, a significant 
# portion (~19%) of requests didn't return any text for the article. After doing some
# individual scraping for some of the URLs that didn't have their text scrapped
# correctly, I found that many of these request can and do return text 
# for the article at the given URL. So, I was most likely sending too many requests 
# to the server while scraping. 

# This script goes through the Scrapped_Text.json file and tries to fix
# these errors. The script will run through the file and if we notice that
# the scrapped text only contains the text of the headline and description of
# the article, the script will try to scrape the text at that url again.
# This time, with no concurrency will be used and the delay between requests will
# be longer to make sure we don't overload the server. 


# the method below is made so we can run it as many times as we want until 
# we see no more successful rescrapes. 



from typing_extensions import final
from constants import *
from save_load_json import load_json, save_json
import time
from web_scraper import get_text_at_link


def make_final_scrapped_set(): 
    article_set = load_json(ARTICLE_SET_PATH)
    scraped_texts = load_json(SCRAPED_TEXT_PATH)
    final_scraped_texts = {category: [] for category in CATEGORIES}

    for category, art_obj_list in article_set.items(): 
        for index, art_obj in enumerate(art_obj_list):
            headline = art_obj['headline']
            description = art_obj['short_description']
            text_check = " ".join(["", headline, description])
            scrapped_text = scraped_texts[category][index]
            if scrapped_text != text_check: 
                final_scraped_texts[category].append(scrapped_text)

    save_json(final_scraped_texts, SCRAPED_TEXT_PATH)
    for category in final_scraped_texts.keys(): 
        print("# of articles in " + category + ' :', len(final_scraped_texts[category]))


def scrape_again(): 
    num_rescraped = 0
    num_successful = 0
    delay = 2
    article_set = load_json(ARTICLE_SET_PATH)
    scraped_texts = load_json(SCRAPED_TEXT_PATH)

    updated_texts = {}
    for category in CATEGORIES: 
        updated_texts[category] = []


    for category, art_obj_list in article_set.items(): 
        for index, art_obj in enumerate(art_obj_list): 

            headline = art_obj['headline']
            description = art_obj['short_description']
            text_check = " ".join(["", headline, description])
            scraped_text = scraped_texts[category][index]

            if scraped_text == text_check: 
                num_rescraped += 1
                print("num_rescraped:", num_rescraped)
                link = art_obj['link']
                updated_text = get_text_at_link(link)
                updated_text = " ".join([updated_text, headline, description])

                if updated_text != text_check: 
                    num_successful += 1
                    print("num_successful:", num_successful)
                    updated_texts[category].append(updated_text)
                else: 
                    updated_texts[category].append(updated_text)
                
                time.sleep(delay)

            else:
                updated_texts[category].append(scraped_text)

    save_json(updated_texts,SCRAPED_TEXT_PATH)
    print("Final Num Rescrapped:", num_rescraped)
    print("Final Num Successful:", num_successful)

if __name__ == "__main__":
    #scrape_again()
    make_final_scrapped_set()

