#Author: Matt Williams
#Version: 6/03/2022


from sklearn.ensemble import BaggingClassifier
from constants import WordVectorModels, ClassificationModels
from run_classification import run_classifier

bagging_parap_grid = {
    "n_estimators" : [],
    "max_samples" : [],
    "max_features" : [],
    "bootstrap" : [True, False], 
    "bootstrap_features" : [True, False],
    "warm_start" : [True, False],
}


def run_bagging(vec_model_name, n_estimators = 10, max_samples = 1.0, max_features = 1.0, \
                bootstrap = True, bootstrap_features = False, warm_start = False): 

    bagging = BaggingClassifier(n_estimators=n_estimators, max_samples=max_samples, max_features=max_features,\
                                bootstrap=bootstrap, bootstrap_features=bootstrap_features, warm_start=warm_start)

    model_details = {
        "Vector_Model" : vec_model_name,
        "Model" : ClassificationModels.BAG.value,
        "N_Estimators" : n_estimators, 
        "Max Samples" : max_samples, 
        "Max Features" : max_features, 
        "Bootstrap" : bootstrap, 
        "Bootstrap_Features" : bootstrap_features, 
        "Warm Start" : warm_start
    }

    run_classifier(vec_model_name, bagging, model_details)



if __name__ == "__main__": 
    run_bagging(WordVectorModels.WORD2VEC.value)
    run_bagging(WordVectorModels.FASTTEXT.value)
    run_bagging(WordVectorModels.GLOVE.value)