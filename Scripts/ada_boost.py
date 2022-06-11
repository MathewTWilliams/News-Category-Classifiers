#Author: Matt Williams
#Version: 6/02/2022

from sklearn.ensemble import AdaBoostClassifier
from utils import WordVectorModels, ClassificationModels
from run_classification import run_classifier

# Parameter grid for cross validation
ada_param_grid = {
    "algorithm" : ['SAMME' , 'SAMME.R'],
    "n_estimators" : list(range(30, 80, 10)),
    "learning_rate" : [1e-4, 1e-3, 1e-2, 1e-1, 1]
}

def run_ada(vec_model_name, algorithm = "SAMME.R", n_estimators = 50, learning_rate = 1.0): 
    '''Given the name of the vector model to train on and the values of the difference hyperparameters, 
    run the Ada-Boost Classification algorithm and save the results to a json file.'''

    ada = AdaBoostClassifier(n_estimators=n_estimators, learning_rate=learning_rate,
                            algorithm= algorithm)

    model_details = {
        "Vector_Model" : vec_model_name, 
        "Model" : ClassificationModels.ADA.value, 
        "Algorithm" : algorithm, 
        "Learning Rate" : learning_rate, 
        "N_Estimators" : n_estimators,
    }

    run_classifier(vec_model_name, ada, model_details)


if __name__ == "__main__": 
    run_ada(WordVectorModels.WORD2VEC.value)
    run_ada(WordVectorModels.FASTTEXT.value)
    run_ada(WordVectorModels.GLOVE.value)
