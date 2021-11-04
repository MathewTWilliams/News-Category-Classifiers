
from nltk import text
from constants import *
from save_load_json import load_json, save_json
from text_cleaner import clean_text
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Pool


def clean_category_texts(category, category_texts):

    cat_cleaned_texts = []
    for index,text in enumerate(category_texts):
        cleaned_text = clean_text(text)
        print("Cleaned a text for: " + category)
        cat_cleaned_texts.append(cleaned_text)
        print(category + " articles cleaned:", index + 1)

    return category, cat_cleaned_texts
  
    



def clean_article_texts(): 
    cleaned_texts = {category: [] for category in CATEGORIES}
    scraped_texts = load_json(SCRAPED_TEXT_PATH)

    with Pool(processes=len(CATEGORIES)) as pool: 

        inputs = zip(CATEGORIES, [scraped_texts[category] for category in CATEGORIES])
        ret_values = pool.starmap(clean_category_texts, inputs )

        for (category, cleaned_texts) in ret_values: 
            cleaned_texts[category] = cleaned_texts

    
    save_json(cleaned_texts, CLEANED_TEXT_PATH)




if __name__ == "__main__":
    clean_article_texts()