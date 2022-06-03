#Author: Matt Williams
#Version: 6/02/2022

from sklearn.ensemble import AdaBoostClassifier
from constants import WordVectorModels
from make_confusion_matrix import show_confusion_matrix
from get_article_vectors import get_test_info, get_training_info
from classifier_metrics import calculate_classifier_metrics

ada_param_grid = {
    "algorithm" : ['SAMME' , 'SAMME.R'],
    "n_estimators" : [30, 40, 50, 60, 70], 
    "learning_rate" : [1e-4, 1e-3, 1e-2, 1e-1, 1]
}


def run_ada(vec_model_name, algorithm = "SAMME.R", n_estimators = 50, learning_rate = 1.0): 
    '''Given the name of the vector model to train on and the values of the difference hyperparameters, 
    run the Ada-Boost Classification algorithm and save the results to a json file.'''


    training_data, training_labels = get_training_info(vec_model_name)
    test_data, test_labels = get_test_info(vec_model_name)

    ada = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                            algorithm= algorithm)

    ada.fit(training_data, training_labels)
    predictions = ada.predict(test_data)

    model_details = {
        "Vector_Model" : vec_model_name, 
        "Model" : "Ada-Boost", 
        "Algorithm" : algorithm, 
        "Learning Rate" : learning_rate, 
        "N_Estimators" : n_estimators,
    }

    calculate_classifier_metrics(test_labels, predictions, model_details)
    show_confusion_matrix(test_labels, predictions, "Ada-Boost w/{} Confusion Matrix".format(vec_model_name))


if __name__ == "__main__": 
     run_ada(WordVectorModels.WORD2VEC.value)
     run_ada(WordVectorModels.FASTTEXT.value)
     run_ada(WordVectorModels.GLOVE.value)