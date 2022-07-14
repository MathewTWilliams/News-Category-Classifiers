#Author: Matt Williams
#Version: 06/24/2022
from sklearn.naive_bayes import ComplementNB
from utils import ClassificationModels, WordVectorModels, CV_BEST_DICT_KEY,  \
    RESULT_WORD_VEC_MOD_KEY, RESULT_MODEL_KEY
from run_classification import run_classifier
from save_load_json import load_cv_result
 
# Parameter grid for cross validation
nb_param_grid = {
    "alpha" : [10 ** i for i in range(-3,1)], 
    "norm" : [True, False]
}
def run_naive_bayes(vec_model_name):

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Gaussian Naive Bayes Classification algorithm and save the results to a json file.'''

    cv_result_dict = load_cv_result(ClassificationModels.CNB.value, vec_model_name)
    best_params_dict = cv_result_dict[CV_BEST_DICT_KEY]
    gauss = ComplementNB(**best_params_dict)
    
    model_details = {
        RESULT_WORD_VEC_MOD_KEY: vec_model_name, 
        RESULT_MODEL_KEY: ClassificationModels.CNB.value,
        CV_BEST_DICT_KEY : best_params_dict
    }

    run_classifier(vec_model_name, gauss, model_details)

