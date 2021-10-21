#Author: Matt Williams
#Version: 10/19/2021

from pathlib import PurePath, Path

# A Simply Python File that contains constant values that are important
#to the project. 

#Dataset: https://www.kaggle.com/rmisra/news-category-dataset 
DATA_SET_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent,
                   "Data/News_Category_Dataset_v2.json").as_posix()

ARTICLE_SET_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent,
                    "Data/Article_Set.json").as_posix()

SORTED_DATA_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                    "Data/Sorted_Data_Set.json").as_posix()

CATEGORIES = ["POLITICS", "WELLNESS", "ENTERTAINMENT", "TRAVEL", 
                "STYLE & BEAUTY", "PARENTING", "HEALTHY LIVING", "QUEER VOICES", 
                "FOOD & DRINK", "BUSINESS", "COMEDY", "SPORTS"]


ARTICLES_PER_CATEGORY = 4800

VALD_SET_SIZE = 480 #10%
TEST_SET_SIZE = 480 #10%
