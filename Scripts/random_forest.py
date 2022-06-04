#Author: Matt Williams
#Version: 12/8/2021

from sklearn.ensemble import RandomForestClassifier
import numpy as np
from constants import RAND_STATE, ClassificationModels, WordVectorModels
from run_classification import run_classifier

#Param Grid for Grid Search Cross Validation
rand_forest_param_grid = {
    'n_estimators' : [int(x) for x in np.linspace(start = 50, stop = 500, num = 5)], 
    'criterion' : ['entropy', 'gini'], 
    'random_state' : [RAND_STATE],
    'max_features' : [None],
    'max_samples' : [(0.1 * i) for i in range(1, 4)]
}



def run_random_forrest(vec_model_name, n_estimators = 100, 
                        criterion = "entropy", max_samples = 1.0): 

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Random Forest Classification algorithm and save the results to a json file.'''


    rf = RandomForestClassifier(criterion= criterion, n_estimators=n_estimators, random_state=RAND_STATE, 
                                max_features=None, max_samples = max_samples)

    model_details = {
        'Vector_Model' : vec_model_name, 
        'Model' : ClassificationModels.RF.value,
        'n_estimators': n_estimators, 
        'criterion': criterion,
        'Max Samples (as fraction)' : max_samples
    }

    run_classifier(vec_model_name, rf, model_details)



if __name__ == "__main__":

    run_random_forrest(WordVectorModels.WORD2VEC.value, n_estimators=500, criterion='entropy', max_samples=0.3)
    run_random_forrest(WordVectorModels.FASTTEXT.value, n_estimators=500, criterion='gini', max_samples=0.3)
    run_random_forrest(WordVectorModels.GLOVE.value, n_estimators=500, criterion='entropy', max_samples=0.3)



    
