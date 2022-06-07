#Author: Matt Williams
#Version: 12/8/2021

from sklearn.svm import SVC
from constants import RAND_STATE, ClassificationModels, WordVectorModels
from run_classification import run_classifier
import numpy as np

#Param Grid for Grid Seartch Cross Validation
svm_param_grid = {
    'C': np.arange(0.2, 1.2, 0.2).tolist(),
    'kernel' : ['linear', 'poly', 'rbf', 'sigmoid'],
    'break_ties' : [True],
    'decision_function_shape' : ['ovr', 'ovo'], 
    'degree' : list(range(2, 7)), 
    'coef0' : np.arange(0.1, 0.6, 0.1).tolist(),
    'tol' : [10 ** i for i in range(-4,1)], 
    'class_weight' : ['balanced']
}

def run_svm(vec_model_name, C = 1.0, kernel = 'rbf', decision_function_shape = 'ovr', \
        degree = 3, coef0 = 0.0, tol = 1e-3):

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Support Vector Machine Classification algorithm and save the results to a json file.'''

    svm = SVC(C=C, kernel=kernel,decision_function_shape=decision_function_shape, random_state=RAND_STATE, \
            break_ties=True, degree=degree, coef0=coef0, class_weight="balanced", tol=tol)

    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : ClassificationModels.SVM.value,
        "C" : C,
        "kernel" : kernel, 
        "Decision Function Shape" : decision_function_shape, 
        "break_ties" : True, 
        "degree" : degree, 
        "coef0" : coef0, 
        "tol" : tol, 
        "class_weights" : "balanced"

    }

    run_classifier(vec_model_name, svm, model_details)



if __name__ == "__main__":
   
   #using best params from grid search cross validation
   run_svm(WordVectorModels.WORD2VEC.value, C=2, kernel="poly", decision_function_shape='ovr')
   run_svm(WordVectorModels.FASTTEXT.value, C=2, kernel='poly', decision_function_shape='ovr')
   run_svm(WordVectorModels.GLOVE.value,C=2, kernel='poly', decision_function_shape='ovr')
