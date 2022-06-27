#Author: Matt Williams
#Version: 06/24/2022

from sklearn.svm import SVC
from utils import ClassificationModels, WordVectorModels, CV_BEST_DICT_KEY
from run_classification import run_classifier
import numpy as np
from save_load_json import load_cv_result

#Parameter grid for cross validation
svm_param_grid = {
    'C': np.arange(0.2, 1.2, 0.2).tolist(),
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'break_ties' : [False],
    'decision_function_shape' : ['ovr', 'ovo'], 
    'degree' : list(range(2, 7)), 
    'coef0' : np.arange(0.1, 0.6, 0.1).tolist(),
    'tol' : [10 ** i for i in range(-4,1)], 
    'class_weight' : ['balanced']
}

def run_svm(vec_model_name):

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Support Vector Machine Classification algorithm and save the results to a json file.'''

    cv_result_dict = load_cv_result(ClassificationModels.SVM.value, vec_model_name)
    best_params_dict = cv_result_dict[CV_BEST_DICT_KEY]


    svm = SVC(**best_params_dict)

    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : ClassificationModels.SVM.value,
        CV_BEST_DICT_KEY : best_params_dict, 
    }

    run_classifier(vec_model_name, svm, model_details)