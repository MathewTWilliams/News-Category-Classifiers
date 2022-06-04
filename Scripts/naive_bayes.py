#Author: Matt Williams
#Version: 12/08/2021
from sklearn.naive_bayes import GaussianNB
from constants import ClassificationModels, WordVectorModels
from run_classification import run_classifier

#Param Grids for Grid Search Cross Validation
gnb_param_grid_1 = {
    'var_smoothing' : [1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13]
}

#From using the previous param grid, we know our best var smoothing value is
#around 1e-4 for all 3 vec models 

gnb_param_grid_2 = {
    'var_smoothing' : [8e-2, 1e-3, 2e-3, 4e-3, 6e-3, 8e-3, 1e-4, 2e-4, 4e-4, 6e-4, 8e-4]
}

#From using the second param grid, we know out best var smoothing value for each
#vec model is 
#Glove: 2e-3
#Word2Vec and FastText: 8e-3

def run_naive_bayes(vec_model_name, var_smoothing = 1e-9):

    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Gaussian Naive Bayes Classification algorithm and save the results to a json file.'''
    gauss = GaussianNB(var_smoothing = var_smoothing)


    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : ClassificationModels.GNB.value,
        'var_smoothing' : var_smoothing
    }

    run_classifier(vec_model_name, gauss, model_details)



if __name__ == "__main__": 

    run_naive_bayes(WordVectorModels.WORD2VEC.value, var_smoothing= 8e-3)
    run_naive_bayes(WordVectorModels.FASTTEXT.value, var_smoothing= 8e-3)
    run_naive_bayes(WordVectorModels.GLOVE.value, var_smoothing = 2e-3)
    
