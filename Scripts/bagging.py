#Author: Matt Williams
#Version: 6/03/2022


from sklearn.ensemble import BaggingClassifier
from utils import WordVectorModels, ClassificationModels
from run_classification import run_classifier

# Parameter Grid for cross validation
bagging_parap_grid = {
    "n_estimators" : list(range(30, 80, 10)),
    "max_samples" : list(range(1000, 16000, 1000)),
    "max_features" : list(range(50, 300, 50)),
    "bootstrap" : [False], 
    "bootstrap_features" : [False]
}


def run_bagging(vec_model_name, n_estimators = 10, max_samples = 1.0, max_features = 1.0): 
    '''Given the name of the vector model to train on and the values of the difference hyperparameters, 
    run the Bagging Classification algorithm and save the results to a json file.'''
    bagging = BaggingClassifier(base_estimator = None, n_estimators=n_estimators, max_samples=max_samples,\
        max_features=max_features,bootstrap=False, bootstrap_features=False, warm_start=False, oob_score=False)

    model_details = {
        "Vector_Model" : vec_model_name,
        "Model" : ClassificationModels.BAG.value,
        "N_Estimators" : n_estimators, 
        "Max Samples" : max_samples, 
        "Max Features" : max_features, 
        "bootstrap" : False, 
        "bootstrap_features" : False
    }

    run_classifier(vec_model_name, bagging, model_details)



if __name__ == "__main__": 
    run_bagging(WordVectorModels.WORD2VEC.value)
    run_bagging(WordVectorModels.FASTTEXT.value)
    run_bagging(WordVectorModels.GLOVE.value)