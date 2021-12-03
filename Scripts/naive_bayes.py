#Author: Matt Williams
#Version: 11/27/2021
from sklearn.naive_bayes import GaussianNB
from get_vec_models import get_vec_model_names
from get_article_vectors import get_training_info, get_test_info
from classifier_metrics import calculate_classifier_metrics

#Param Grids for Grid Search Cross Validation
gnb_param_grid_1 = {
    'var_smoothing' : [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
}

#From using the previous param grid, we know our best var smoothing value is
#around 1e-4 for all 3 vec models (first 3 results in CV_Results folder)

gnb_param_grid_2 = {
    'var_smoothing' : [8e-2, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4]
}

#From using the second param grid, we know out best var smoothing value for each
#vec model is (from last 3 results in CV_Results Folder): 
#Glove: 2e-3
#Word2Vec and FastText: 8e-3

def run_naive_bayes(vec_model_name, var_smoothing = 1e-9):

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Gaussian Naive Bayes Classification algorithm and save the results to a json file.'''
    training_data, training_labels = get_training_info(vec_model_name)
    test_data, test_labels = get_test_info(vec_model_name)
    gauss = GaussianNB(var_smoothing = var_smoothing)

    gauss.fit(training_data, training_labels)
    predictions = gauss.predict(test_data)

    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : "Gaussian_NB",
        'var_smoothing' : var_smoothing
    }

    calculate_classifier_metrics(test_labels, predictions, model_details)



if __name__ == "__main__": 

    run_naive_bayes("word2vec", var_smoothing= 8e-3)
    run_naive_bayes("fasttext", var_smoothing= 8e-3)
    run_naive_bayes("glove", var_smoothing = 2e-3)
    