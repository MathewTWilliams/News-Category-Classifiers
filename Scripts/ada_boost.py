#Author: Matt Williams
#Version: 6/24/2022

from sklearn.ensemble import AdaBoostClassifier
from utils import WordVectorModels, ClassificationModels, CV_BEST_DICT_KEY
from run_classification import run_classifier
from save_load_json import load_cv_result

# Parameter grid for cross validation
ada_param_grid = {
    "algorithm" : ['SAMME' , 'SAMME.R'],
    "n_estimators" : list(range(30, 80, 10)),
    "learning_rate" : [1e-4, 1e-3, 1e-2, 1e-1, 1]
}

def run_ada(vec_model_name): 
    '''Given the name of the vector model to train on and the values of the difference hyperparameters, 
    run the Ada-Boost Classification algorithm and save the results to a json file.'''

    cv_results_dict = load_cv_result(ClassificationModels.ADA.value, vec_model_name)
    best_params_dict = cv_results_dict[CV_BEST_DICT_KEY]
    ada = AdaBoostClassifier()

    model_details = {
        "Vector_Model" : vec_model_name, 
        "Model" : ClassificationModels.ADA.value, 
        CV_BEST_DICT_KEY: best_params_dict,

    }

    run_classifier(vec_model_name, ada, model_details)


if __name__ == "__main__": 
    run_ada(WordVectorModels.WORD2VEC.value)
    run_ada(WordVectorModels.FASTTEXT.value)
    run_ada(WordVectorModels.GLOVE.value)
