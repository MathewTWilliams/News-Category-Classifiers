#Author: Matt Williams
#Version: 11/7/2021

from pathlib import PurePath, Path

# A Simple Python File that contains constant values that are important
# to the Preprocessing Aspect of the Project. 

#Dataset: https://www.kaggle.com/rmisra/news-category-dataset 

#This file contains the original data set. 
DATA_SET_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent,
                   "Data/News_Category_Dataset_v2.json").as_posix()

#This file contains the original data set sorted by article category
SORTED_DATA_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                    "Data/Sorted_Data_Set.json").as_posix()

#This file contains the articles to be used for classification
ARTICLE_SET_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent,
                    "Data/Article_Set.json").as_posix()

#This file contains some text to test text_cleaner.py on before cleaning 
#all the articles. 
TEST_TEXT_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                    "Data/test_text.txt").as_posix()

#This file contains a dictionary where each key is a category and each value a list of strings, 
# where each string is the scrapped text of an article. 
SCRAPED_TEXT_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                    "Data/Scraped_Text.json").as_posix()      

# This file contains the cleaned text from the Scrapped Text file.
# Contains a dictionary where each key is a category and each value is a 3d list: 
# dict[category][i][j][k] - kth word in jth sentence in ith article
CLEANED_TEXT_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent,
                    "Data/Cleaned_Text.json").as_posix()              

# List of categories to be used from the original dataset. 
CATEGORIES = ["MEDIA", "WEIRD NEWS", "GREEN", "WORLDPOST",
                "RELIGION", "STYLE", "SCIENCE", "WORLD NEWS",
                "TASTE", "TECH"]


VALID_SET_PERC = 0.1
TEST_SET_PERC = 0.1
TRAIN_SET_PERC = 0.8
TTS_RAND_STATE = 42


MODEL_DIR_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, "Models/")


def get_w2v_path(name): 
    return PurePath.joinpath(MODEL_DIR_PATH, name).as_posix()


ARTICLE_VECS_DIR_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, "Article_Vectors/")

def get_article_vecs_path(name):
    return PurePath.joinpath(ARTICLE_VECS_DIR_PATH, name).as_posix()