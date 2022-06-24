#Author: Matt Williams
#Version: 6/24/2022


from sklearn.ensemble import GradientBoostingClassifier
from utils import ClassificationModels, WordVectorModels, CV_BEST_DICT_KEY
from run_classification import run_classifier
import numpy as np
from save_load_json import load_cv_result

# Parameter grid for cross validation
grad_boost_param_grid = {
    "loss" : ["log_loss", "exponential"],
    "learning_rate" : [10**i for i in range(-4,1)], 
    "n_estimators" : list(range(80,130, 10)), 
    "subsample" : np.arange(0.2, 1.1, 0.2).tolist(),
    "criterion" : ["friedman_mse", "squared_error"],
    "min_weight_fraction_leaf" : np.arange(0.1, 0.6, 0.1).tolist(),
    "max_depth" : list(range(3,8)), 
    "max_features" : ["sqrt", "log2"],
    "min_impurity_decrease" : [0.0001],
    "n_iter_no_change" : [1],
    "tol" : [10**i for i in range(-4, -1)]

}

def run_grad_boost(vec_model_name ): 
    '''Given the name of the vector model to train on and the values of the difference hyperparameters, 
    run the Gradient Boost Classification algorithm and save the results to a json file.'''



    cv_results_dict = load_cv_result(ClassificationModels.GRAD.value, vec_model_name)
    best_params_dict = cv_results_dict[CV_BEST_DICT_KEY]

    grad_boost = GradientBoostingClassifier(**best_params_dict)


    model_details = {
        "Vector_Model" : vec_model_name, 
        "Model" : ClassificationModels.GRAD.value,
        CV_BEST_DICT_KEY : best_params_dict, 
    }

    run_classifier(vec_model_name, grad_boost, model_details)


if __name__ == "__main__": 
    run_grad_boost(WordVectorModels.WORD2VEC.value)
    run_grad_boost(WordVectorModels.FASTTEXT.value)
    run_grad_boost(WordVectorModels.GLOVE.value)
