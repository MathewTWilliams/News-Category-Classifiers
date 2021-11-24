from get_vec_models import get_model_names
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from get_article_vectors import get_training_info, get_test_info
from classifier_metrics import calculate_classifier_metrics
#Parameters for Random Search Cross Validation. Cross validation ran with random combinations
#of hyerparameters. Used to help condense our search for the best hyperparameters to use.
num_estimators = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 5)]
criterions = ["gini", "entropy"]
max_depths = [int(x) for x in np.linspace(start = 10, stop = 100, num = 10)]
min_samples_splits = [int(x) for x in np.linspace(start = 2, stop = 10, num = 5)]
min_samples_leaves = [int(x) for x in np.linspace(start = 1, stop = 5, num = 5)]
max_nodes = [int(x) for x in np.linspace(start = 50, stop = 100, num = 5)]


rf_random_grid = {
    'n_estimators': num_estimators, 
    'criterion': criterions,
    'max_depth': max_depths, 
    'min_samples_split': min_samples_splits, 
    'min_samples_leaf': min_samples_leaves, 
    'max_leaf_nodes': max_nodes
}




#Parameters for grid Cross Validation. Cross validation is ran with each
#possible combination of hyper parameters
#These parameters are chosen after running Random Search Cross Validation






def run_random_forrest(model_name, n_estimators = 100, 
                        criterion = "gini", max_depth = None, min_samples_split = 2, 
                        min_samples_leaf = 1, max_leaf_nodes = None ): 

    training_data, training_labels = get_training_info(model_name)
    test_data, test_labels = get_test_info(model_name)


    rf = RandomForestClassifier(criterion= criterion, n_estimators=n_estimators, 
                                max_depth=max_depth, min_samples_split=min_samples_split, 
                                min_samples_leaf=min_samples_leaf, max_leaf_nodes=max_leaf_nodes)

    rf.fit(training_data, training_labels)
    predictions = rf.predict(test_data)


    model_details = {
        'Vector_model' : model_name, 
        'Classifier' : "Random_Forest",
        'n_estimators': n_estimators, 
        'criterion': criterion,
        'max_depth': "None" if max_depth == None else str(max_depth), 
        'min_sample_split': min_samples_split, 
        'min_sample_leaf': min_samples_leaves, 
        'max_leaf_nodes': "None" if max_leaf_nodes == None else str(max_leaf_nodes)
    }

    calculate_classifier_metrics(test_labels, predictions, model_details)




if __name__ == "__main__":

    model_name = get_model_names()[0]
    run_random_forrest(model_name)


    
