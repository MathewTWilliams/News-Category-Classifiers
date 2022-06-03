#Author: Matt Williams
#Version: 12/02/2021

import os
from gensim.models import KeyedVectors
from constants import MODEL_DIR_PATH, get_model_path, WordVectorModels

def get_vec_models(): 
    """Returns a list of word2vec gensim models found in the models folder."""
    model_wvs = {}

    name, wvs = get_w2v_model()
    model_wvs[name] = wvs
    name, wvs = get_fasttext_model()
    model_wvs[name] = wvs
    name, wvs = get_glove_model()
    model_wvs[name] = wvs

    return model_wvs

def get_vec_model(model_name): 
    '''Given the filename of a model, return the name of the model without the file extention, 
    and the WordVectors associated with that model.'''
    if not os.path.exists(MODEL_DIR_PATH):
        return None

    path = get_model_path(model_name)
    return model_name.split('.')[0], KeyedVectors.load(path)

def get_w2v_model(): 
    '''Returns Word2Vec Model's WordVectors'''
    return get_vec_model("word2vec.model")

def get_fasttext_model(): 
    '''Returns the Fast Test Model's WordVectors'''
    return get_vec_model("fasttext.model")

def get_glove_model(): 
    '''Returns the Glove Model's WordVectors'''
    return get_vec_model("glove.model")

def get_vec_model_names(): 
    '''Returns a list of names of the models used in this project, without the file extention.'''
    return WordVectorModels.get_values_as_list()


if __name__ == "__main__":
    #models = get_vec_models()
    #print(len(models.items()))
    print(get_vec_model_names())
