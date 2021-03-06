#Author: Matt Williams
#Version: 6/24/2022

from sklearn.neighbors import RadiusNeighborsClassifier
from utils import ClassificationModels, WordVectorModels, CV_BEST_DICT_KEY,  \
    RESULT_WORD_VEC_MOD_KEY, RESULT_MODEL_KEY
from run_classification import run_classifier
import numpy as np
from save_load_json import load_cv_result

# Parameter grid for cross validation
near_rad_param_grid = {
    "radius" : np.arange(10, 51, 10).tolist(), #here
    "algorithm" : ["ball_tree", "kd_tree"], 
    "leaf_size" : list(range(10, 60, 10)), 
    "p" : list(range(1,5)), 
    "metric" : ["minkowski"], 
    "weights" : ["distance"]
}


def run_near_radius(vec_model_name): 
    
    cv_result_dict = load_cv_result(ClassificationModels.RAD.value, vec_model_name)
    best_params_dict = cv_result_dict[CV_BEST_DICT_KEY]
    best_params_dict['weights'] = "distance"
    near_radius = RadiusNeighborsClassifier(**best_params_dict)

    model_details = {
        RESULT_WORD_VEC_MOD_KEY : vec_model_name, 
        RESULT_MODEL_KEY : ClassificationModels.RAD.value, 
        CV_BEST_DICT_KEY : best_params_dict, 
    }

    run_classifier(vec_model_name, near_radius, model_details)
