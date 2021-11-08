#Auther: Matt Williams
#Version: 11/7/2021

from constants import *
from save_load_json import load_json, save_json
from text_cleaner import clean_text
from multiprocessing import Pool


def clean_category_texts(category, category_texts):
    """Given a category and a list of article texts, clean each article text. 
        Returns the category given and a 3D list where: 
        cat_clean_texts[i][j][k] is the kth word in the jth sentence in the ith article"""
    cat_cleaned_texts = []
    for index,text in enumerate(category_texts):
        cleaned_text_mat = clean_text(text)
        print("Cleaned a text for: " + category)
        cat_cleaned_texts.append(cleaned_text_mat)
        print(category + " articles cleaned:", index + 1)


    return category, cat_cleaned_texts
  
    



def clean_article_texts(): 
    """Main method for this script. Load in our scraped texts, then use
        multiprocessing with the method above to help clean the articles faster. 
        Once all articles are cleaned, save the cleaned texts to a new json file"""
    cleaned_texts_dict = {category: [] for category in CATEGORIES}
    scraped_texts = load_json(SCRAPED_TEXT_PATH)

    with Pool(processes=len(CATEGORIES)) as pool: 

        inputs = zip(CATEGORIES, [scraped_texts[category] for category in CATEGORIES])
        ret_values = pool.starmap(clean_category_texts, inputs )

        for (category, cleaned_texts) in ret_values: 
            cleaned_texts_dict[category] = cleaned_texts

    
    save_json(cleaned_texts_dict, CLEANED_TEXT_PATH)




if __name__ == "__main__":
    clean_article_texts()