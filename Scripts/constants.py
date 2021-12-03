#Author: Matt Williams
#Version: 12/02/2021

from pathlib import PurePath, Path
import os
import pandas as pd
import numpy as np


# A Simple Python File that contains constant values and methods important to the project. 
# Most of these values are file path related. 

#Dataset: https://www.kaggle.com/rmisra/news-category-dataset 

#This file contains the original data set. 
DATA_SET_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent,
                   "Data/News_Category_Dataset_v2.json").as_posix()

#This file contains the original data set sorted by article category
SORTED_DATA_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                    "Data/Sorted_Data_Set.json").as_posix()

#This file contains the articles to be used for the project.
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
# dict[category][i][j][k] - kth word in jth sentence in ith article for the given category
CLEANED_TEXT_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                    "Data/Cleaned_Text.json").as_posix()              

# List of categories to be used from the original dataset. 
CATEGORIES = ["MEDIA", "WEIRD NEWS", "GREEN", "WORLDPOST",
                "RELIGION", "STYLE", "SCIENCE", "WORLD NEWS",
                "TASTE", "TECH"]


TEST_SET_PERC = 0.1
TRAIN_SET_PERC = 0.9
RAND_STATE = 42
K_FOLDS = 10

#File Path to the directory that contains our Models related to generating word embeddings. 
MODEL_DIR_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, "Models/")


def get_model_path(name): 
    '''Given the name of a file for a vector model, return the full file path for that file to be stored at.'''
    if not os.path.exists(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    return PurePath.joinpath(MODEL_DIR_PATH, name).as_posix()


#File Path to the directory where the article vectors are to be stored.
ARTICLE_VECS_DIR_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, "Article_Vectors/")

def get_article_vecs_path(subfolder, name):
    '''Given subfolder(Train or Test) and the name of a file, return the full file path for that file to be stored at'''
    if not os.path.exists(ARTICLE_VECS_DIR_PATH):
        os.mkdir(ARTICLE_VECS_DIR_PATH)
    return PurePath.joinpath(ARTICLE_VECS_DIR_PATH, subfolder).joinpath(name).as_posix()

#File Path to the directory where classification or clustering results are to be stored.
RESULTS_DIR_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                                                "Results/")

def make_result_path(subfolder, vec_model): 
    '''Given the name of a subfolder(Name of the Classification/Clustering algorithm) and the name
        of the vector model, generate a name for the file and return the full file path for the file to be stored at.'''

    if not os.path.exists(RESULTS_DIR_PATH):
        os.mkdir(RESULTS_DIR_PATH)

    updated_path = PurePath.joinpath(RESULTS_DIR_PATH, subfolder)

    if not os.path.exists(updated_path): 
        os.mkdir(updated_path)

    file_name = vec_model + "_results_" + str(len(os.listdir(updated_path)) + 1) + ".json"
    return updated_path.joinpath(file_name).as_posix()

def get_result_path(subfolder, file_name): 
    '''Get the full file path for the result with the given filename and the given subfolder.'''
    return PurePath.joinpath(RESULTS_DIR_PATH, subfolder).joinpath(file_name).as_posix()

#File Path to the directory where cross validation results are to be stored.
CV_RESULTS_DIR_PATH = PurePath.joinpath(Path(__file__).resolve().parent.parent, 
                                                "CV_Results/")

def make_cv_result_path(subfolder, vec_model): 
    '''Given the name of a subfolder(Name of the Classification/Clustering algorithm), generate a name 
        for the file and return the full file path for the file to be stored at. '''
    if not os.path.exists(CV_RESULTS_DIR_PATH): 
        os.mkdir(CV_RESULTS_DIR_PATH)

    updated_path = PurePath.joinpath(CV_RESULTS_DIR_PATH, subfolder)

    if not os.path.exists(updated_path): 
        os.mkdir(updated_path)

    file_name = vec_model + "_cv_results_" + str(len(os.listdir(updated_path)) + 1) + ".json"
    return updated_path.joinpath(file_name).as_posix()


def get_cv_result_path(subfolder, file_name): 
    '''Get the full file path for the cross validation result with the given filename in the given subfolder.'''
    return PurePath.joinpath(CV_RESULTS_DIR_PATH, subfolder).joinpath(file_name).as_posix()

def convert_categories_to_numbers(labels):
    '''given a pandas series that contains the different categories, convert those 
    categories into integers. Integers represent location in CATEGORIES list.'''

    if isinstance(labels, pd.Series):
        for i,category in enumerate(CATEGORIES):
            labels = labels.replace(category, i)
    
    elif isinstance(labels, np.ndarray):
        for i,category in enumerate(CATEGORIES): 
            labels[labels == category] = i

    return labels