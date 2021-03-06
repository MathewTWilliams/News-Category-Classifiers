#Author: Matt Williams
#Version: 06/24/2022
from sklearn.neighbors import KNeighborsClassifier
from utils import WordVectorModels, ClassificationModels, CV_BEST_DICT_KEY,  \
    RESULT_WORD_VEC_MOD_KEY, RESULT_MODEL_KEY
from run_classification import run_classifier
from save_load_json import load_cv_result

#Parameter grid for Cross Validation
knn_param_grid = {
    "n_neighbors" : list(range(5, 30, 5)),
    "weights" : ['uniform', 'distance'], 
    "algorithm" : ['ball_tree', 'kd_tree'],
    "p" : list(range(1,5)),
    "leaf_size" : list(range(10, 60, 10))

}


def run_knn(vec_model_name):
    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the K-Nearest Neighbors Classification algorithm and save the results to a json file.'''
    
    cv_results_dict = load_cv_result(ClassificationModels.KNN.value, vec_model_name)
    best_params_dict = cv_results_dict[CV_BEST_DICT_KEY]
    knn = KNeighborsClassifier(**best_params_dict)
                                
    model_details = {
        RESULT_WORD_VEC_MOD_KEY: vec_model_name, 
        RESULT_MODEL_KEY : ClassificationModels.KNN.value,
        CV_BEST_DICT_KEY : best_params_dict, 
    }

    run_classifier(vec_model_name, knn, model_details)
        
    