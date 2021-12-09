#Author: Matt Williams
#Version: 12/5/2021

from knn import knn_param_grid
from logistic_regression import log_regr_param_grid
from ml_perceptron import mlp_param_grid
from naive_bayes import gnb_param_grid_1, gnb_param_grid_2
from random_forest import rand_forest_param_grid
from svm import svm_param_grid

from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression

from cross_validation import run_grid_cv

from get_vec_models import get_vec_model_names




if __name__ == "__main__": 
    '''run grid search validation for each classifier for each vector model'''
    for name in get_vec_model_names(): 
        run_grid_cv(SVC(), svm_param_grid, name, "Support Vector Machine")
        run_grid_cv(RandomForestClassifier(), rand_forest_param_grid, name, "Random Forest")
        run_grid_cv(MLPClassifier(), mlp_param_grid, name, "Multi-Layer Perceptron")
        run_grid_cv(KNeighborsClassifier(), knn_param_grid, name, "K-Nearest Neighbor")
        run_grid_cv(GaussianNB(), gnb_param_grid_2, name, "Gaussian Naive Bayes")
        run_grid_cv(LogisticRegression(), log_regr_param_grid, name, "Logistic Regression")


