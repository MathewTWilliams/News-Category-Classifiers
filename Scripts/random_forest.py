#Author: Matt Williams
#Version: 12/8/2021

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from constants import RAND_STATE, ClassificationModels, WordVectorModels
from run_classification import run_classifier

#Parameter grid for cross validation
rand_forest_param_grid = {
    'n_estimators' : list(range(80,130, 10)), 
    'criterion' : ['entropy', 'gini', "log_loss"], 
    'random_state' : [RAND_STATE],
    'max_features' : [None, "sqrt", "log2"],
    'max_samples' : np.arange(0.4, 0.9, 0.1).tolist(),
    'min_weight_fraction_leaf' : np.arange(0.1,0.6,0.1).tolist(),
    "min_impurity_decrease" : [0.0001],
    "class_weight" : ["balanced"]
}
def run_random_forrest(vec_model_name, n_estimators = 100, criterion = "entropy", max_samples = 1.0, \
                    max_features = "sqrt", min_weight_fraction_leaf = 0.0, min_impurity_decrease = 0.0 ): 

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Random Forest Classification algorithm and save the results to a json file.'''


    rf = RandomForestClassifier(criterion= criterion, n_estimators=n_estimators, random_state=RAND_STATE, \
                            max_features = max_features, max_samples = max_samples, \
                            min_weight_fraction_leaf=min_weight_fraction_leaf, min_impurity_decrease=min_impurity_decrease, \
                            class_weight="balanced")

    model_details = {
        'Vector_Model' : vec_model_name, 
        'Model' : ClassificationModels.RF.value,
        'n_estimators': n_estimators, 
        'criterion': criterion,
        'Max Samples' : max_samples, 
        'Max Features' : max_features, 
        'min_weight_fraction_leaf' : min_weight_fraction_leaf, 
        'min_impurity_decrease' : min_impurity_decrease,
        'class_weight' : "balanced"
    }

    run_classifier(vec_model_name, rf, model_details)



if __name__ == "__main__":

    run_random_forrest(WordVectorModels.WORD2VEC.value, n_estimators=500, criterion='entropy', max_samples=0.3)
    run_random_forrest(WordVectorModels.FASTTEXT.value, n_estimators=500, criterion='gini', max_samples=0.3)
    run_random_forrest(WordVectorModels.GLOVE.value, n_estimators=500, criterion='entropy', max_samples=0.3)



    
