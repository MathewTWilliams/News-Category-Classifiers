#Author: Matt Williams
#Version: 11/7/2021


from bs4 import BeautifulSoup
from Scripts.constants import Categories
from save_load_json import save_json, load_json
from constants import ARTICLE_SET_PATH, SCRAPED_TEXT_PATH
from http import HTTPStatus
import requests
from concurrent.futures import ThreadPoolExecutor
import time

def clean_html(html): 
    """Given the html of a huffpost webpage, grab the text body of the article and return 
    it as a string."""
    soup = BeautifulSoup(html, "html.parser")
    
    #in HuffPost articles, any div with the class ="primary-cli cli cli-text"   
    #contains a portion of the article body. 
    content = soup.find_all("div", 
                        {"class": "primary-cli cli cli-text"})

    text = " ".join([child.get_text() for child in content])

    #remove any extra newline characters
    text = text.translate(text.maketrans("\n\t\r", "   "))
    return text



def check_if_page_exists(url): 
    """Given a huffpost url, check to make sure the article at that url is still reachable."""
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
    """Method used by other scripts, used to obtain the article text at the given link."""
    return clean_html(requests.get(link).text)

def scrape_category_list(**kwargs):
    """Given a list of keywords arguements which should contain the time delay, article_objects, and category
    Scrape the text found at the link in the article_object. Then the text is combined with the headline and description
    found in the article object. This combined object is then saved into a list. THe method returns this list and the category."""
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
    """The main method for this script. Load in the Article Set json object, then use multi-threading and
    the method above in order to scrape the articles for each category. Then save the scrapped text to a json file."""
    article_set = load_json(ARTICLE_SET_PATH)
    delay = 1.5
    scraped_text = {}
    futures = []

    categories = Categories.get_values_as_list()
    with ThreadPoolExecutor(max_workers=len(categories)) as executor: 

        for category in categories: 
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

    save_json(scraped_text, SCRAPED_TEXT_PATH)

if __name__ == "__main__":
    scrape_articles()

    
        











    