#Author: Matt Williams
#Version: 12/5/2021

from knn import knn_param_grid
from logistic_regression import log_regr_param_grid
from naive_bayes import nb_param_grid
from random_forest import rand_forest_param_grid
from svm import svm_param_grid
from ada_boost import ada_param_grid
from bagging import bagging_parap_grid
from near_centroid import near_cent_param_grid
from near_radius import near_rad_param_grid
from mlp import mlp_param_grid
from grad_boost import grad_boost_param_grid


from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, BaggingClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from cross_validation import run_grid_cv

from get_vec_models import get_vec_model_names
from utils import ClassificationModels



if __name__ == "__main__": 
    '''run grid search validation for each classifier for each vector model'''
    for name in get_vec_model_names(): 
        run_grid_cv(SVC(), svm_param_grid, name, ClassificationModels.SVM.value)
        run_grid_cv(RandomForestClassifier(), rand_forest_param_grid, name, ClassificationModels.RF.value)
        run_grid_cv(KNeighborsClassifier(), knn_param_grid, name, ClassificationModels.KNN.value)
        run_grid_cv(ComplementNB(), nb_param_grid, name, ClassificationModels.CNB.value)
        run_grid_cv(LogisticRegression(), log_regr_param_grid, name, ClassificationModels.LOG_REG.value)
        run_grid_cv(AdaBoostClassifier(), ada_param_grid, name, ClassificationModels.ADA.value)
        run_grid_cv(MLPClassifier(), mlp_param_grid, name, ClassificationModels.MLP.value)
        run_grid_cv(BaggingClassifier(), bagging_parap_grid, name, ClassificationModels.BAG.value)
        run_grid_cv(NearestCentroid(), near_cent_param_grid, name, ClassificationModels.CENT.value)  
        #run_grid_cv(GradientBoostingClassifier(), grad_boost_param_grid, ClassificationModels.GRAD.value)      
        #run_grid_cv(RadiusNeighborsClassifier(), near_rad_param_grid, name,  ClassificationModels.RAD.value)


