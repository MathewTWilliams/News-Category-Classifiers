#Author: Matt Williams
#Version: 12/8/2021
from sklearn.svm import SVC
from get_article_vectors import get_test_info, get_training_info
from classifier_metrics import calculate_classifier_metrics
from constants import RAND_STATE
from make_confusion_matrix import show_confusion_matrix

#Param Grid for Grid Seartch Cross Validation
svm_param_grid = {
    'C': [0.25, 0.5, 1, 1.5, 2], 
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'break_ties' : [True],
    'decision_function_shape' : ['ovr', 'ovo']
}

def run_svm(vec_model_name, C = 1.0, kernel = 'rbf',
              decision_function_shape = 'ovr'):

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Support Vector Machine Classification algorithm and save the results to a json file.'''

    training_data, training_labels = get_training_info(vec_model_name)
    test_data, test_labels = get_test_info(vec_model_name)

    svm = SVC(C=C, kernel=kernel,decision_function_shape=decision_function_shape,
                random_state=RAND_STATE, break_ties=True)


    svm.fit(training_data, training_labels)
    predictions = svm.predict(test_data)

    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : "Support Vector Machine",
        "C" : C,
        "kernel" : kernel, 
        "Decision Function Shape" : decision_function_shape 
    }

    calculate_classifier_metrics(test_labels, predictions, model_details)
    show_confusion_matrix(test_labels, predictions, "Support Vector Machine w/" + vec_model_name + " Confusion Matrix")



if __name__ == "__main__":
   
   #using best params from grid search cross validation
   run_svm("word2vec", C=2, kernel="poly", decision_function_shape='ovr')
   run_svm("fasttext", C=2, kernel='poly', decision_function_shape='ovr')
   run_svm("glove" ,C=2, kernel='poly', decision_function_shape='ovr')
