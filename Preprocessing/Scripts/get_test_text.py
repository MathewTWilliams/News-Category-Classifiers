#Author: Matt Williams
#Version: 11/7/2021


import requests
from web_scraper import clean_html
from constants import TEST_TEXT_PATH
import os 

def get_test_text(): 
    """Using a random article url from our dataset, scrape and save the text to a 
    file for us to use later to test our text_cleaner.py"""
    url = "https://www.huffpost.com/entry/texas-amanda-painter-mass-shooting_n_5b081ab4e4b0802d69caad89"
    html = requests.get(url).text
    text = clean_html(html)

    if os.path.exists(TEST_TEXT_PATH): 
        os.remove(TEST_TEXT_PATH)

    with open(TEST_TEXT_PATH, "w+", encoding = "utf-8") as file:
        file.write(text) 




if __name__ == "__main__":
    get_test_text()