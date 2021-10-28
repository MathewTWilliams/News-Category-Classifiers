import requests

from text_cleaner import clean_html
from constants import TEST_TEXT_PATH


def get_test_text(): 
    url = "https://www.huffpost.com/entry/texas-amanda-painter-mass-shooting_n_5b081ab4e4b0802d69caad89"
    html = requests.get(url).text
    text = clean_html(html)
    with open(TEST_TEXT_PATH, "w+", encoding = "utf-8") as file:
        file.write(text) 




if __name__ == "__main__":
    get_test_text()