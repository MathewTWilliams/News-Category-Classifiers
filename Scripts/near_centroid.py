#Author: Matt Williams
#Version: 6/24/2022

from sklearn.neighbors import NearestCentroid
from utils import ClassificationModels, WordVectorModels, CV_BEST_DICT_KEY,  \
    RESULT_WORD_VEC_MOD_KEY, RESULT_MODEL_KEY
from run_classification import run_classifier
import numpy as np
from save_load_json import load_cv_result

# Parameter grid for cross validation
near_cent_param_grid = {
    "metric" : ["cosine", "euclidean", "manhattan"], 
    "shrink_threshold" : np.arange(0.1, 0.6, 0.1).tolist()
}


def run_near_centroid(vec_model_name): 
    '''Given the name of the vector model to train on and the values of the difference hyperparameters, 
    run the Nearest Centroid Classification algorithm and save the results to a json file.'''
    cv_result_dict = load_cv_result(ClassificationModels.CENT.value, vec_model_name)
    best_params_dict = cv_result_dict[CV_BEST_DICT_KEY]
    near_cent = NearestCentroid(**best_params_dict)

    model_details = {
        RESULT_WORD_VEC_MOD_KEY : vec_model_name, 
        RESULT_MODEL_KEY : ClassificationModels.CENT.value, 
        CV_BEST_DICT_KEY : best_params_dict
    }

    run_classifier(vec_model_name, near_cent, model_details)
