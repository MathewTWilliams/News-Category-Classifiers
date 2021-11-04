#Author: Matt Williams
#Version: 10/19/2021

from pathlib import PurePath, Path

# A Simply Python File that contains constant values that are important
#to the project. 

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

#This file contains a category and a list of strings, where each string is 
#is the scrapped text of an article. 
SCRAPED_TEXT_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                    "Data/Scraped_Text.json").as_posix()      

# This file contains the cleaned text from the Scrapped Text file
CLEANED_TEXT_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent,
                    "Data/Cleaned_Text.json")              




CATEGORIES = ["MEDIA", "WEIRD NEWS", "GREEN", "WORLDPOST",
                "RELIGION", "STYLE", "SCIENCE", "WORLD NEWS",
                "TASTE", "TECH"]


VALID_SET_PERC = 10
TEST_SET_PERC = 10
TRAIN_SET_PERC = 80
