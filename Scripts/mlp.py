#Author: Matt Williams
#Version: 6/04/2022


from sklearn.neural_network import MLPClassifier
from utils import ClassificationModels, WordVectorModels
from run_classification import run_classifier
import numpy as np

# Parameter grid for cross validation
mlp_param_grid = {
    "hidden_layer_sizes" : [(10, ), (10, 10,), (10,10,5)], 
    "solver" : ["lbfgs", "sgd", "adam"], 
    "activation" : ['logistic', 'tanh', 'relu'], 
    "alpha" : [10**i for i in range(-4, 2)], 
    "learning_rate" : ["constant", 'invscaling', 'adaptive'], 
    "learning_rate_init" : [10**i for i in range(-4, 2)], 
    "max_iter" : [500], 

    "early_stopping" : [True],  

}

def run_mlp(vec_model_name, ): 
    '''Given the name of the vector model to train on and the values of the difference hyperparameters, 
    run the Gradient Boost Classification algorithm and save the results to a json file.'''
    mlp = MLPClassifier()


    model_details = {
        "Vector_Model" : vec_model_name, 
        "Model" : ClassificationModels.MLP.value,

    }

    run_classifier(vec_model_name, mlp, model_details)


if __name__ == "__main__": 
    run_mlp(WordVectorModels.WORD2VEC.value)
    run_mlp(WordVectorModels.FASTTEXT.value)
    run_mlp(WordVectorModels.GLOVE.value)
