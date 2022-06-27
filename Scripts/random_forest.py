#Author: Matt Williams
#Version: 6/24/2022

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from utils import ClassificationModels, WordVectorModels, CV_BEST_DICT_KEY
from run_classification import run_classifier
from save_load_json import load_cv_result

#Parameter grid for cross validation
rand_forest_param_grid = {
    'n_estimators' : list(range(80,130, 10)), 
    'criterion' : ['entropy', 'gini', "log_loss"], 
    'max_features' : [None, "sqrt", "log2"],
    'max_samples' : np.arange(0.4, 0.9, 0.1).tolist(),
    'min_weight_fraction_leaf' : np.arange(0.1,0.6,0.1).tolist(),
    "min_impurity_decrease" : [0.0001],
    "class_weight" : ["balanced"]
}
def run_random_forest(vec_model_name): 
    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Random Forest Classification algorithm and save the results to a json file.'''

    cv_result_dict = load_cv_result(ClassificationModels.RF.value, vec_model_name)
    best_params_dict = cv_result_dict[CV_BEST_DICT_KEY]

    rf = RandomForestClassifier(**best_params_dict)

    model_details = {
        'Vector_Model' : vec_model_name, 
        'Model' : ClassificationModels.RF.value,
        CV_BEST_DICT_KEY : best_params_dict
    }

    run_classifier(vec_model_name, rf, model_details)




    
