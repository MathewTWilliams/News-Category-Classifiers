#Author: Matt Williams
#Version: 12/08/2021
from sklearn.naive_bayes import ComplementNB
from utils import ClassificationModels, WordVectorModels
from run_classification import run_classifier
 
# Parameter grid for cross validation
nb_param_grid = {
    "alpha" : [10 ** i for i in range(-3,1)], 
    "norm" : [True, False]
}
def run_naive_bayes(vec_model_name, alpha = 1.0, norm = False):

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Gaussian Naive Bayes Classification algorithm and save the results to a json file.'''
    gauss = ComplementNB(alpha = alpha, norm = norm)


    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : ClassificationModels.CNB.value,
        'alpha' : alpha, 
        'norm' : norm, 
    }

    run_classifier(vec_model_name, gauss, model_details)



if __name__ == "__main__": 

    run_naive_bayes(WordVectorModels.WORD2VEC.value, var_smoothing= 8e-3)
    run_naive_bayes(WordVectorModels.FASTTEXT.value, var_smoothing= 8e-3)
    run_naive_bayes(WordVectorModels.GLOVE.value, var_smoothing = 2e-3)
    
