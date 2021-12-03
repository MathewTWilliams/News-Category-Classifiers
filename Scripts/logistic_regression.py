
#Author: Matt Williams
#Version: 12/02/021
from get_vec_models import get_vec_model_names
from sklearn.linear_model import LogisticRegression
from get_article_vectors import get_test_info, get_training_info
from classifier_metrics import calculate_classifier_metrics
from constants import RAND_STATE

#Param Grids for Grid Search Cross Validation
log_regr_param_grid_1 = {
    'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'tol' : [1e-3, 1e-4, 1e-5], 
    'C' : [0.5, 1.0, 2.0], 
    'solver' : ['saga'], 
    'max_iter' : [100, 200, 500],
    'random_state' : [RAND_STATE]
}

log_regr_param_grid_2 = {
    'penalty' : ['l2'],
    'tol' : [1e-3, 1e-4, 1e-5], 
    'C' : [0.5, 1.0, 2.0], 
    'solver' : ['saga', 'sag', 'lbfgs', 'newton-cg'], 
    'max_iter' : [100, 200, 500],
    'random_state' : [RAND_STATE]
}

def run_logistic_regression(vec_model_name, penalty = 'l2', tol = 1e-4, C = 1,
                            solver = 'lbfgs',  max_iter = 100):

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Logistic Regression Classification algorithm and save the results to a json file.'''
    training_data, training_labels = get_training_info(vec_model_name)
    test_data, test_labels = get_test_info(vec_model_name)

    weights_dict = {}
    for category in training_labels: 
        if category not in weights_dict.keys():
            weights_dict[category] = 1
        else: 
            weights_dict[category] += 1

    for category in weights_dict.keys(): 
        weights_dict[category] /= len(training_labels)
    


    lr = LogisticRegression(penalty=penalty, random_state=RAND_STATE, tol= tol, solver=solver, 
                            class_weight= weights_dict, C=C, max_iter=max_iter)
    lr.fit(training_data, training_labels)
    predictions = lr.predict(test_data)


    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : "Logistic_Regression",
        'Penalty_Function' : penalty, 
        'tolerance' : tol, 
        'Solver': solver,
        'C' : C,
        'Max Iterations': max_iter

    }

    calculate_classifier_metrics(test_labels, predictions, model_details)



if __name__ == "__main__": 
    vec_model_name = get_vec_model_names()[2]
    run_logistic_regression(vec_model_name)