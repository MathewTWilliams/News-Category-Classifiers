#Author: Matt Williams
#Version: 11/27/2021
from get_vec_models import get_vec_model_names
from sklearn.ensemble import RandomForestClassifier
from get_article_vectors import get_training_info, get_test_info
from classifier_metrics import calculate_classifier_metrics
import numpy as np
from constants import RAND_STATE


#Param Grid for Grid Search Cross Validation
rand_forest_param_grid = {
    'n_estimators' : [int(x) for x in np.linspace(start = 50, stop = 500, num = 5)], 
    'criterion' : ['entropy', 'gini'], 
    'random_state' : [RAND_STATE],
    'max_features' : [None],
    'max_samples' : [(0.1 * i) for i in range(1, 7)]
}



def run_random_forrest(vec_model_name, n_estimators = 100, 
                        criterion = "entropy", max_samples = 1.0): 

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Random Forest Classification algorithm and save the results to a json file.'''
    training_data, training_labels = get_training_info(vec_model_name)
    test_data, test_labels = get_test_info(vec_model_name)


    rf = RandomForestClassifier(criterion= criterion, n_estimators=n_estimators, random_state=RAND_STATE, 
                                max_features=None, max_samples = max_samples)

    rf.fit(training_data, training_labels)
    predictions = rf.predict(test_data)


    model_details = {
        'Vector_Model' : vec_model_name, 
        'Model' : "Random_Forest",
        'n_estimators': n_estimators, 
        'criterion': criterion,
        'Max Samples (as fraction)' : max_samples
    }

    calculate_classifier_metrics(test_labels, predictions, model_details)




if __name__ == "__main__":

    vec_model_name = get_vec_model_names()[1]
    run_random_forrest(vec_model_name)




    
