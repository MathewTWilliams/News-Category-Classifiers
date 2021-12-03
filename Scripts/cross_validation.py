#Author: Matt Williams
#Version: 12/02/2021

#Grid Search Cross Validation is done so we can get a better understanding
#of the hyperparameter settings we need in order to optimize the algorithms for
#our dataset. 

#Reference: https://towardsdatascience.com/hyperparameter-tuning-the-random-forest-in-python-using-scikit-learn-28d2aa77dd74


from numpy import ndarray
from sklearn.model_selection import GridSearchCV, cross_validate
from get_article_vectors import get_training_info
from save_load_json import save_json
from constants import make_cv_result_path, K_FOLDS

#deprecated: grid search cross validation was taking too much time. 
"""def run_grid_cv(classifier, param_grid, vec_model_name, c_name, n_jobs = 3):
    '''Given a classifier instance, its associated param grid, the name of the vector model
    we are to use the training data from, and the classifier name. Run Grid Search Cross validation and
    save the results to a json file. '''
    training_data, training_labels = get_training_info(vec_model_name)
    grid_search_cv = GridSearchCV(classifier, param_grid, n_jobs=n_jobs, verbose=2, cv = 10, refit='accuracy',
                                    scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro']) 

    grid_search_cv.fit(training_data, training_labels)

    file_path = get_cv_result_path(c_name, vec_model_name)
    cv_results = grid_search_cv.cv_results_
    for key in cv_results.keys(): 
        if isinstance(cv_results[key], ndarray): 
            cv_results[key] = cv_results[key].tolist()
    cv_results['best params'] = grid_search_cv.best_params_
    cv_results['Word Vector Model'] = vec_model_name
    save_json(cv_results, file_path)"""


def run_cross_validation(classifier, classifier_name, vec_model_name):
    '''Given a classifier instance, the name of the vector model
    we are to use the training data from, and the classifier name, run Cross validation and
    save the results to a json file.'''
    training_data, training_labels = get_training_info(vec_model_name)

    cv_results = cross_validate(classifier, training_data, training_labels, cv=K_FOLDS, return_train_score=True,  
                                scoring=['accuracy', 'precision_macro', 'recall_macro', 'f1_macro'])

    file_path = make_cv_result_path(classifier_name, vec_model_name)
    for key in cv_results.keys(): 
        if isinstance(cv_results[key], ndarray): 
            cv_results[key] = cv_results[key].tolist()
    cv_results['Word Vector Model'] = vec_model_name
    cv_results['Params'] = classifier.get_params(False)
    save_json(cv_results, file_path)

    