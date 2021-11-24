#Author: Matt Williams
#Version: 11/21/2021
from constants import ARTICLE_VECS_DIR_PATH, get_article_vecs_path
import os 
import pandas as pd
import numpy as np


def get_article_vectors(model_name, set):
    '''Given a model name and the name of the set wanted (Train or Test), 
    return the data and labels associated with the given set of the given model.'''
    if not os.path.exists(ARTICLE_VECS_DIR_PATH): 
        return None, None

    if set not in os.listdir(ARTICLE_VECS_DIR_PATH):
        return None, None

    for file in os.listdir(get_article_vecs_path(set, "")):
        if file.startswith(model_name) and file.find("article_vecs") == -1:

                vector_set = pd.read_json(get_article_vecs_path(set, file))
                return vector_set.drop("Category", axis=1), vector_set["Category"]

    return None, None


def get_training_info(model_name):
    '''Given a model name, this method returns the training data and training labels as separate values.'''
    return get_article_vectors(model_name, set = "Train")

def get_test_info(model_name):
    '''Given a model name, this method returns the test data and test labels as separate values.'''
    return get_article_vectors(model_name, set = "Test")

    
if __name__ == "__main__":
    data, labels = get_training_info("word2vec")

    r_index = np.isnan(data)
    for r in r_index:
        print(r)
