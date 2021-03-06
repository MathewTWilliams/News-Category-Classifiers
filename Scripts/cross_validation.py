#Author: Matt Williams
#Version: 06/24/2022

#Grid Search Cross Validation is done so we can get a better understanding
#of the hyperparameter settings we need in order to optimize the algorithms for
#our dataset. 

from numpy import ndarray
from sklearn.experimental import enable_halving_search_cv
from sklearn.model_selection import HalvingGridSearchCV
from sklearn.naive_bayes import ComplementNB
from sklearn.preprocessing import minmax_scale
from get_article_vectors import get_training_info
from save_load_json import save_json
from utils import make_cv_result_path, K_FOLDS, CV_BEST_DICT_KEY

def run_grid_cv(classifier, param_grid, vec_model_name, c_name, n_jobs = 3):
    '''Given a classifier instance, its associated param grid, the name of the vector model
    we are to use the training data from, and the classifier name. Run Grid Search Cross validation and
    save the results to a json file. '''

    training_data, training_labels = get_training_info(vec_model_name)
    grid_search_cv = HalvingGridSearchCV(classifier, param_grid, n_jobs=n_jobs, verbose=2, cv = K_FOLDS, refit=False,
                                    scoring='accuracy', min_resources=100) 



    if type(classifier) is ComplementNB: 
        training_data = minmax_scale(training_data, feature_range=(0,1))

    grid_search_cv.fit(training_data, training_labels)

    file_path = make_cv_result_path(c_name, vec_model_name)
    cv_results = grid_search_cv.cv_results_
    for key in cv_results.keys(): 
        if isinstance(cv_results[key], ndarray): 
            cv_results[key] = cv_results[key].tolist()
    cv_results[CV_BEST_DICT_KEY] = grid_search_cv.best_params_
    cv_results['Word Vector Model'] = vec_model_name
    cv_results['best score'] = grid_search_cv.best_score_
    save_json(cv_results, file_path)



    