#Author: Matt Williams
#Version: 10.14.2021



from urllib.request import urlopen
import json
from constants import ARTICLE_SET_PATH

def load_article_set(): 
    
    with open(ARTICLE_SET_PATH, "r+") as file: 
        aritcle_set_map = json.load(file)
        for cat, article_maps in aritcle_set_map.items(): 
            for article_map in article_maps:
                page = urlopen(article_map['link'])
                html_bytes = page.read()
                html = html_bytes.decode("utf-8")

if __name__ == "__main__": 
    