#Author: Matt Williams
#Version: 10.15.2021


from bs4 import BeautifulSoup
from constants import ARTICLE_SET_PATH
from save_load_json import load_json
from http import HTTPStatus
import requests


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

    
def test_page_exists():
    bad_url = "https://www.huffingtonpost.com/entry/kaiser-carlile-dead_us_55bf5973e4b06363d5a2988e"
    good_url = "https://www.huffpost.com/entry/donald-trump-mcondalds-tonight-show_n_5b093561e4b0fdb2aa53daba"
    exists_redirect_url = "https://www.huffingtonpost.com/entry/thanksgiving-space-nasa-atronauts-iss-video_us_5baebc8ee4b014374e2eb14e"

    print(check_if_page_exists(bad_url))
    print(check_if_page_exists(good_url))
    print(check_if_page_exists(exists_redirect_url))




def scrape_articles():
    article_set = load_json(ARTICLE_SET_PATH)

    for category, article_list in article_set.items(): 
        for url in article_list: 
            html = requests.get(url).text
            text = clean_html(html)
            



if __name__ == "__main__":
    test_page_exists()











    