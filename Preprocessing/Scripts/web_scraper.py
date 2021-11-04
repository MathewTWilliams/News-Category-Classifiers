#Author: Matt Williams
#Version: 10.31.2021


from bs4 import BeautifulSoup
from requests.api import request
from constants import CATEGORIES, SCRAPPED_TEXT_PATH
from save_load_json import save_json
from constants import ARTICLE_SET_PATH
from save_load_json import load_json
from http import HTTPStatus
import requests
from concurrent.futures import ThreadPoolExecutor
import time



#in HuffPost articles, any div with the class ="primary-cli cli cli-text"
#contains a portion of the article body. 
def clean_html(html): 

    soup = BeautifulSoup(html, "html.parser")
    content = soup.find_all("div", 
                        {"class": "primary-cli cli cli-text"})

    text = " ".join([child.get_text() for child in content])

    #remove any extra newline characters
    text = text.translate(text.maketrans("\n\t\r", "   "))
    return text



def check_if_page_exists(url): 

    url_check = "https://www.huffpost.com/section"

    try: 
        response = requests.head(url)
        if response.status_code == HTTPStatus.OK:
            return True
        elif response.status_code == HTTPStatus.MOVED_PERMANENTLY: 
            location = response.headers['Location']
            return not location.startswith(url_check)
    except: 
        return False


def get_text_at_link(link): 
    return clean_html(requests.get(link).text)

def scrape_category_list(**kwargs):
    
    article_objects = kwargs["article_objects"]
    delay = kwargs["delay"]
    category = kwargs["category"]
    scraped_text_list = []

    
    for art_obj in article_objects: 
        headline = art_obj['headline']
        link = art_obj['link']
        description = art_obj['short_description']
        text = get_text_at_link(link)
        text = " ".join([text, headline, description])
        scraped_text_list.append(text)
        print(category, " articles scrapped:", len(scraped_text_list))
        time.sleep(delay)

    return category, scraped_text_list


def scrape_articles(): 
    article_set = load_json(ARTICLE_SET_PATH)
    delay = 1.5
    scraped_text = {}
    futures = []

    with ThreadPoolExecutor(max_workers=len(CATEGORIES)) as executor: 

        for category in CATEGORIES: 
            scraped_text[category] = []
            article_objects = article_set[category]
            future = executor.submit(scrape_category_list, 
                                    category = category, 
                                    article_objects = article_objects, 
                                    delay = delay)
            futures.append(future)

    for future in futures: 
        category, scrapped_text_list = future.result()
        scraped_text[category] = scrapped_text_list

    save_json(scraped_text, SCRAPPED_TEXT_PATH)

if __name__ == "__main__":
    scrape_articles()

    
        











    