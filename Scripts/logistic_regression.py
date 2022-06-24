
#Author: Matt Williams
#Version: 06/24/2022
from sklearn.linear_model import LogisticRegression
from get_article_vectors import get_training_info
from utils import WordVectorModels, ClassificationModels, CV_BEST_DICT_KEY
from run_classification import run_classifier
import numpy as np
from save_load_json import load_cv_result

#Parameter grid for cross validation
log_regr_param_grid= {
    'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'tol' : [10**i for i in range(-4, 1)], 
    'C' : np.arange(0.2,1.2,0.2).tolist(),
    'solver' : ['saga'], 
    'max_iter' : list(range(100, 600, 100)),
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
    
    cv_results = load_cv_result(ClassificationModels.LOG_REG.value, vec_model_name)
    best_params_dict = cv_results[CV_BEST_DICT_KEY]

    lr = LogisticRegression(class_weight= weights_dict, **best_params_dict)


    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : ClassificationModels.LOG_REG.value,
        CV_BEST_DICT_KEY : best_params_dict, 

    }

    run_classifier(vec_model_name, lr, model_details)


if __name__ == "__main__": 
    
    run_logistic_regression(WordVectorModels.WORD2VEC.value)
    run_logistic_regression(WordVectorModels.FASTTEXT.value)
    run_logistic_regression(WordVectorModels.GLOVE.value)