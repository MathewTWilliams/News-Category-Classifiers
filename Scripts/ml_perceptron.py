#Author: Matt Williams
#Version: 12/08/2021
from sklearn.neural_network import MLPClassifier
from constants import RAND_STATE, ClassificationModels, WordVectorModels
from run_classification import run_classifier

#Param Grid for Grid Search Cross Validation
mlp_param_grid = {
    'hidden_layer_sizes': [(10,), (10,10)], 
    'activation' : ['identity', 'logistic', 'tanh', 'relu'],
    'solver' : ['adam'],
    'alpha' : [1e-3, 1e-4, 1e-5], 
    'early_stopping' : [True],
    'validation_fraction' : [0.1],
    'random_state' : [RAND_STATE]

}


def run_mlp(vec_model_name, hl_sizes = (100,), activation = "relu", solver = 'adam', 
                alpha = 1e-4): 
    '''Given the name of the vector model to train on and the values of the different hyperparameters, 
    run the Multi-Layer Perceptron Classification algorithm and save the results to a json file.'''


    mlp = MLPClassifier(hidden_layer_sizes=hl_sizes, activation=activation, solver=solver,
                        alpha=alpha, random_state=RAND_STATE, early_stopping=True, validation_fraction=0.1)


    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : ClassificationModels.MLP.value,
        "Hidden Layer Sizes" : hl_sizes ,
        "activation" : activation, 
        "solver" : solver,
        "alpha" : alpha, 
    }

    run_classifier(vec_model_name, mlp, model_details)
    
if __name__ == "__main__": 
    
    run_mlp(WordVectorModels.WORD2VEC.value, hl_sizes=(10,), activation='tanh', solver='adam', alpha=1e-3)
    run_mlp(WordVectorModels.FASTTEXT.value, hl_sizes=(10,), activation='tanh', solver='adam', alpha=1e-3)
    run_mlp(WordVectorModels.GLOVE.value, hl_sizes=(10,), activation='tanh', solver='adam', alpha=1e-3)