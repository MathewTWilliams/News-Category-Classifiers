
#Author: Matt Williams
#Version: 12/08/021
from sklearn.linear_model import LogisticRegression
from get_article_vectors import get_training_info
from utils import RAND_STATE, WordVectorModels, ClassificationModels
from run_classification import run_classifier
import numpy as np

#Parameter grid for cross validation
log_regr_param_grid= {
    'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'tol' : [10**i for i in range(-4, 1)], 
    'C' : np.arange(0.2,1.2,0.2).tolist(),
    'solver' : ['saga'], 
    'max_iter' : list(range(100, 600, 100)),
    'random_state' : [RAND_STATE], 
    'multi_class' : ["ovr", "multinomial"], 
    'l1_ratio' : np.arange(0, 1.25, 0.25)
}

def run_logistic_regression(vec_model_name, penalty = 'l2', tol = 1e-4, C = 1,
                            solver = 'saga',  max_iter = 100, multi_class = "multinomial"):

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Logistic Regression Classification algorithm and save the results to a json file.'''
    _, training_labels = get_training_info(vec_model_name)

    weights_dict = {}
    for category in training_labels: 
        if category not in weights_dict.keys():
            weights_dict[category] = 1
        else: 
            weights_dict[category] += 1

    for category in weights_dict.keys(): 
        weights_dict[category] /= len(training_labels)
    


    lr = LogisticRegression(penalty=penalty, random_state=RAND_STATE, tol= tol, solver=solver, 
                            class_weight= weights_dict, C=C, max_iter=max_iter, multi_class=multi_class)


    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : ClassificationModels.LOG_REG.value,
        'Penalty_Function' : penalty, 
        'tolerance' : tol, 
        'Solver': solver,
        'C' : C,
        'Max Iterations': max_iter, 
        "multi_class" : multi_class

    }

    run_classifier(vec_model_name, lr, model_details)


if __name__ == "__main__": 
    
    run_logistic_regression(WordVectorModels.WORD2VEC.value, penalty="l1", tol=1e-3, C=2, solver = "saga", max_iter=200)
    run_logistic_regression(WordVectorModels.FASTTEXT.value, penalty="none", tol=1e-3, C=0.5, solver = "saga", max_iter=100)
    run_logistic_regression(WordVectorModels.GLOVE.value, penalty='l2', tol=1e-4, C=2, max_iter=100)