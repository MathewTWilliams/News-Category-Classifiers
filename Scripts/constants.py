#Author: Matt Williams
#Version: 12/08/2021

from enum import Enum
import os
import pandas as pd
import numpy as np
import re

# A Simple Python File that contains constant values and methods important to the project. 
# Most of these values are file path related. 


class Datasets(Enum): 
    TRAIN = "Train"
    TEST = "Test"

class WordVectorModels(Enum): 
    FASTTEXT = "fasttext"
    GLOVE = "glove"
    WORD2VEC = "word2vec"

    @classmethod
    def get_values_as_list(self): 
        return [model.value for model in WordVectorModels]

class ClassificationModels(Enum): 
    KNN = "K-Nearest Neighbor"
    SVM = "Support Vector Machine"
    RF = "Random Forest"
    LOG_REG = "Logistic Regression"
    ADA = "Ada-Boost"
    BAG = "Bagging"
    GRD_BST = "Gradient Boosting"
    CENT = "Nearest Centroid"
    RAD = "Radius Neighbors"
    CNB = "Complement Naive Bayes"


    @classmethod
    def get_values_as_list(self): 
        return [model.value for model in ClassificationModels]

class Categories(Enum): 
    MEDIA = "MEDIA"
    WEIRD = "WEIRD NEWS"
    GREEN =  "GREEN"
    POST =  "WORLDPOST"
    RELIGION = "RELIGION"
    STYLE =  "STYLE"
    SCIENCE =  "SCIENCE"
    NEWS =  "WORLD NEWS"
    TASTE = "TASTE"
    TECH =  "TECH"

    @classmethod
    def get_values_as_list(self): 
        return [cat.value for cat in Categories]

#Dataset: https://www.kaggle.com/rmisra/news-category-dataset 
CWD_PATH = os.path.abspath(os.getcwd())
DATA_DIR_PATH = os.path.join(CWD_PATH, "Data")

#This file contains the original data set. 
DATA_SET_PATH = os.path.join(DATA_DIR_PATH, "News_Category_Dataset_v2.json")

#This file contains the original data set sorted by article category
SORTED_DATA_PATH = os.path.join(DATA_DIR_PATH, "Sorted_Data_Set.json")

#This file contains the articles to be used for the project.
ARTICLE_SET_PATH = os.path.join(DATA_DIR_PATH, "Article_Set.json")

#This file contains some text to test text_cleaner.py on before cleaning 
#all the articles. 
TEST_TEXT_PATH = os.path.join(DATA_DIR_PATH, "test_text.txt")

#This file contains a dictionary where each key is a category and each value a list of strings, 
# where each string is the scrapped text of an article.   
SCRAPED_TEXT_PATH = os.path.join(DATA_DIR_PATH, "Scraped_Text.json")


# This file contains the cleaned text from the Scrapped Text file.
# Contains a dictionary where each key is a category and each value is a 3d list: 
# dict[category][i][j][k] - kth word in jth sentence in ith article for the given category
CLEANED_TEXT_PATH = os.path.join(DATA_DIR_PATH, "Cleaned_Text.json")      


TEST_SET_PERC = 0.1
TRAIN_SET_PERC = 0.9
RAND_STATE = 42
K_FOLDS = 10


#File Path to the directory that contains our Models related to generating word embeddings. 
MODEL_DIR_PATH = os.path.join(CWD_PATH, "Models")

def get_model_path(name): 
    '''Given the name of a file for a vector model, return the full file path for that file to be stored at.'''
    if not os.path.exists(MODEL_DIR_PATH):
        os.mkdir(MODEL_DIR_PATH)
    return os.path.join(MODEL_DIR_PATH, name)


#File Path to the directory where the article vectors are to be stored.
ARTICLE_VECS_DIR_PATH = os.path.join(CWD_PATH, "Article_Vectors")

def get_article_vecs_path(subfolder, name):
    '''Given subfolder(Train or Test) and the name of a file, return the full file path for that file to be stored at'''
    if not os.path.exists(ARTICLE_VECS_DIR_PATH):
        os.mkdir(ARTICLE_VECS_DIR_PATH)
    res_path = os.path.join(ARTICLE_VECS_DIR_PATH, subfolder)
    return os.path.join(res_path, name)

#File Path to the directory where classification or clustering results are to be stored.
RESULTS_DIR_PATH = os.path.join(CWD_PATH, "Results")

def find_json_file_name_number(folder):
    '''Given a file path to a folder containing .json files, find the highest number associated with a filename in that
    folder, then increment that number by 1 and return it '''
    highest_num = 0
    num_pattern = re.compile(r"[0-9]+\.json")
    for file in os.listdir(folder):
        match = num_pattern.search(file).group()
        num = int(match.split(".")[0])
        if num > highest_num: 
            highest_num = num
    
    return highest_num + 1
        

def make_result_path(subfolder, vec_model): 
    '''Given the name of a subfolder(Name of the Classification/Clustering algorithm) and the name
        of the vector model, generate a name for the file and return the full file path for the file to be stored at.'''

    if not os.path.exists(RESULTS_DIR_PATH):
        os.mkdir(RESULTS_DIR_PATH)

    updated_path = os.path.join(RESULTS_DIR_PATH, subfolder)

    if not os.path.exists(updated_path): 
        os.mkdir(updated_path)

    num = find_json_file_name_number(updated_path)
    file_name = vec_model + "_results_{}.json".format(num)
    return os.path.join(updated_path, file_name)


def get_result_path(subfolder, file_name): 
    '''Get the full file path for the result with the given filename and the given subfolder.'''
    res_path = os.path.join(RESULTS_DIR_PATH, subfolder)
    return os.path.join(res_path, file_name)

#File Path to the directory where cross validation results are to be stored.
CV_RESULTS_DIR_PATH = os.path.join(CWD_PATH,"CV_Results")

def make_cv_result_path(subfolder, vec_model): 
    '''Given the name of a subfolder(Name of the Classification/Clustering algorithm), generate a name 
        for the file and return the full file path for the file to be stored at. '''
    if not os.path.exists(CV_RESULTS_DIR_PATH): 
        os.mkdir(CV_RESULTS_DIR_PATH)

    updated_path = os.path.join(CV_RESULTS_DIR_PATH, subfolder)

    if not os.path.exists(updated_path): 
        os.mkdir(updated_path)

    num = find_json_file_name_number(updated_path)
    file_name = vec_model + "_cv_results_{}.json".format(num) 
    return os.path.join(updated_path, file_name)


def get_cv_result_path(subfolder, file_name): 
    '''Get the full file path for the cross validation result with the given filename in the given subfolder.'''
    res_path = os.path.join(CV_RESULTS_DIR_PATH, subfolder)
    return os.path.join(res_path, file_name)

def convert_categories_to_numbers(labels):
    '''given a pandas series that contains the different categories, convert those 
    categories into integers. Integers represent location in CATEGORIES list.'''

    if isinstance(labels, pd.Series):
        for i,category in enumerate(Categories.get_values_as_list()):
            labels = labels.replace(category, i)
    
    elif isinstance(labels, np.ndarray):
        for i,category in enumerate(Categories.get_values_as_list()): 
            labels[labels == category] = i

    return labels
