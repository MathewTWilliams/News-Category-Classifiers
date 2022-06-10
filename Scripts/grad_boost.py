#Author: Matt Williams
#Version: 6/04/2022


from sklearn.ensemble import GradientBoostingClassifier
from constants import ClassificationModels, WordVectorModels
from run_classification import run_classifier
import numpy as np

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

def run_grad_boost(vec_model_name, loss = "log_loss", learning_rate = 0.1, n_estimators = 100,\
                subsample = 1.0, criterion = "friedman_mse", min_weight_fraction_leaf = 0.0,\
                max_depth = 3, max_features = None, min_impurity_decrease = 0.0, n_iter_no_change = None,\
                tol = 1e-4): 
    '''Given the name of the vector model to train on and the values of the difference hyperparameters, 
    run the Gradient Boost Classification algorithm and save the results to a json file.'''
    grad_boost = GradientBoostingClassifier(loss=loss, learning_rate=learning_rate, n_estimators=n_estimators, \
                                        subsample=subsample, criterion=criterion, min_weight_fraction_leaf=min_weight_fraction_leaf, \
                                        max_depth=max_depth, max_features=max_features, min_impurity_decrease=min_impurity_decrease, \
                                        n_iter_no_change=n_iter_no_change, tol=tol)


    model_details = {
        "Vector_Model" : vec_model_name, 
        "Model" : ClassificationModels.GRD_BST.value,
        "loss" : loss, 
        "learning_rate" : learning_rate, 
        "n_estimators" : n_estimators, 
        "subsample" : subsample, 
        "criterion" : criterion, 
        "min_weight_fraction_leaf" : min_weight_fraction_leaf, 
        "max_depth" : max_depth, 
        "max_features" : max_features, 
        "min_impurity_decrease" : min_impurity_decrease, 
        "n_iter_no_change" : n_iter_no_change, 
        "tol" : tol
    }

    run_classifier(vec_model_name, grad_boost, model_details)


if __name__ == "__main__": 
    run_grad_boost(WordVectorModels.WORD2VEC.value)
    run_grad_boost(WordVectorModels.FASTTEXT.value)
    run_grad_boost(WordVectorModels.GLOVE.value)
