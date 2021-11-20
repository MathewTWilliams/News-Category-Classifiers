
from constants import ARTICLE_VECS_DIR_PATH, get_article_vecs_path
import os 
import pandas as pd


def get_article_vectors(model_name, set = "Train"):

    if not os.path.exists(ARTICLE_VECS_DIR_PATH): 
        return None

    if set not in os.listdir(ARTICLE_VECS_DIR_PATH):
        return None


    combined_df = pd.DataFrame()
    for file in os.listdir(ARTICLE_VECS_DIR_PATH + "/" + set):
        if file.startswith(model_name):
                category_df = pd.read_json(get_article_vecs_path(file), typ='frame')
                combined_df = combined_df.append(category_df)

    return combined_df


def get_training_vectors(model_name):
    return get_article_vectors(model_name)

def get_validation_vectors(model_name):
    return get_article_vectors(model_name, set = "Valid")

def get_test_vectors(model_name):
    return get_article_vectors(model_name, set = "Test")

    


if __name__ == "__main__":
    print(get_training_vectors("CBOW").columns)