#Author: Matt Williams
#Version: 12/08/2021
from sklearn.neural_network import MLPClassifier
from get_article_vectors import get_test_info, get_training_info
from classifier_metrics import calculate_classifier_metrics
from make_confusion_matrix import show_confusion_matrix
from constants import RAND_STATE, ClassificationModels, WordVectorModels

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

    training_data, training_labels = get_training_info(vec_model_name)
    test_data, test_labels = get_test_info(vec_model_name)

    mlp = MLPClassifier(hidden_layer_sizes=hl_sizes, activation=activation, solver=solver,
                        alpha=alpha, random_state=RAND_STATE, early_stopping=True, validation_fraction=0.1)

    mlp.fit(training_data, training_labels)
    predictions = mlp.predict(test_data)


    model_details = {
        'Vector_Model': vec_model_name, 
        'Model' : ClassificationModels.MLP.value,
        "Hidden Layer Sizes" : hl_sizes ,
        "activation" : activation, 
        "solver" : solver,
        "alpha" : alpha, 
    }

    calculate_classifier_metrics(test_labels, predictions, model_details)
    show_confusion_matrix(test_labels, predictions, "ML-Perceptron w/" + vec_model_name + " Confusion Matrix")
    
if __name__ == "__main__": 
    
    run_mlp(WordVectorModels.WORD2VEC.value, hl_sizes=(10,), activation='tanh', solver='adam', alpha=1e-3)
    run_mlp(WordVectorModels.FASTTEXT.value, hl_sizes=(10,), activation='tanh', solver='adam', alpha=1e-3)
    run_mlp(WordVectorModels.GLOVE.value, hl_sizes=(10,), activation='tanh', solver='adam', alpha=1e-3)