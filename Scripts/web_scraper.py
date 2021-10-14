#Author: Matt Williams
#Version: 10.15.2021



from urllib.request import urlopen
from constants import ARTICLE_SET_PATH
from load_dataset import load_dataset
from text_cleaner import clean_html

def scrape_articles():
    article_set = load_dataset(ARTICLE_SET_PATH)

    for category, article_list in article_set.items(): 
        html = urlopen(article_list[0]).read().decode("utf-8")
        text = clean_html(html)
        print(text)


    
    
            





if __name__ == "__main__":
    article_set_map = scrape_articles()












    