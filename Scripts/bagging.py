#Author: Matt Williams
#Version: 6/24/2022


from msilib.schema import Class
from sklearn.ensemble import BaggingClassifier
from utils import WordVectorModels, ClassificationModels, CV_BEST_DICT_KEY
from run_classification import run_classifier
import numpy as np
from save_load_json import load_cv_result

# Parameter Grid for cross validation
bagging_parap_grid = {
    "n_estimators" : list(range(30, 80, 10)),
    "max_samples" : np.arange(0.1, 1.1, 0.1).tolist(),
    "max_features" : list(range(50, 300, 50)),
    "bootstrap" : [False], 
    "bootstrap_features" : [False]
}


def run_bagging(vec_model_name): 


    '''Given the name of the vector model to train on and the values of the difference hyperparameters, 
    run the Bagging Classification algorithm and save the results to a json file.'''

    cv_results_dict = load_cv_result(ClassificationModels.BAG.value, vec_model_name)
    best_params_dict = cv_results_dict[CV_BEST_DICT_KEY]
    best_params_dict["base_estimator"] = None
    best_params_dict["warm_start"] = False
    best_params_dict["oob_score"] = False
    bagging = BaggingClassifier(**best_params_dict)

    model_details = {
        "Vector_Model" : vec_model_name,
        "Model" : ClassificationModels.BAG.value,
        CV_BEST_DICT_KEY : best_params_dict, 
    }

    run_classifier(vec_model_name, bagging, model_details)



if __name__ == "__main__": 
    run_bagging(WordVectorModels.WORD2VEC.value)
    run_bagging(WordVectorModels.FASTTEXT.value)
    run_bagging(WordVectorModels.GLOVE.value)