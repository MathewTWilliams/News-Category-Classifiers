
#Author: Matt Williams
#Version: 12/08/021
from sklearn.linear_model import LogisticRegression
from get_article_vectors import get_test_info, get_training_info
from classifier_metrics import calculate_classifier_metrics
from constants import RAND_STATE, WordVectorModels, ClassificationModels
from make_confusion_matrix import show_confusion_matrix

#Param Grids for Grid Search Cross Validation
log_regr_param_grid= {
    'penalty' : ['l1', 'l2', 'elasticnet', 'none'],
    'tol' : [1e-3, 1e-4, 1e-5], 
    'C' : [0.5, 1.0, 2.0], 
    'solver' : ['saga'], 
    'max_iter' : [100, 200, 300],
    'random_state' : [RAND_STATE]
}

def run_logistic_regression(vec_model_name, penalty = 'l2', tol = 1e-4, C = 1,
                            solver = 'saga',  max_iter = 100):

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
        'Model' : ClassificationModels.LOG_REG.value,
        'Penalty_Function' : penalty, 
        'tolerance' : tol, 
        'Solver': solver,
        'C' : C,
        'Max Iterations': max_iter

    }

    calculate_classifier_metrics(test_labels, predictions, model_details)
    show_confusion_matrix(test_labels, predictions, "Logistic Regression w/" + vec_model_name + " Confusion Matrix")


if __name__ == "__main__": 
    
    run_logistic_regression(WordVectorModels.WORD2VEC.value, penalty="l1", tol=1e-3, C=2, solver = "saga", max_iter=200)
    run_logistic_regression(WordVectorModels.FASTTEXT.value, penalty="none", tol=1e-3, C=0.5, solver = "saga", max_iter=100)
    run_logistic_regression(WordVectorModels.GLOVE.value, penalty='l2', tol=1e-4, C=2, max_iter=100)