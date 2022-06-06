#Author: Matt Williams
#Version: 12/5/2021

from knn import knn_param_grid
from logistic_regression import log_regr_param_grid
from naive_bayes import nb_param_grid
from random_forest import rand_forest_param_grid
from svm import svm_param_grid
from ada_boost import ada_param_grid
from bagging import bagging_parap_grid
from grad_boost import grad_boost_param_grid
from near_centroid import near_cent_param_grid
from near_radius import near_rad_param_grid



from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier, BaggingClassifier
from sklearn.neighbors import KNeighborsClassifier, RadiusNeighborsClassifier, NearestCentroid
from sklearn.naive_bayes import ComplementNB
from sklearn.linear_model import LogisticRegression

from cross_validation import run_grid_cv

from get_vec_models import get_vec_model_names




if __name__ == "__main__": 
    '''run grid search validation for each classifier for each vector model'''
    for name in get_vec_model_names(): 
        run_grid_cv(SVC(), svm_param_grid, name, "Support Vector Machine")
        run_grid_cv(RandomForestClassifier(), rand_forest_param_grid, name, "Random Forest")
        run_grid_cv(KNeighborsClassifier(), knn_param_grid, name, "K-Nearest Neighbor")
        run_grid_cv(ComplementNB(), nb_param_grid, name, "Complement Naive Bayes")
        run_grid_cv(LogisticRegression(), log_regr_param_grid, name, "Logistic Regression")
        run_grid_cv(AdaBoostClassifier, ada_param_grid, "Ada-Boost")
        run_grid_cv(GradientBoostingClassifier, grad_boost_param_grid, "Gradient Boosted Decision Trees")
        run_grid_cv(BaggingClassifier, bagging_parap_grid, "Bagging")
        run_grid_cv(RadiusNeighborsClassifier, near_rad_param_grid, "Near Radius")
        run_grid_cv(NearestCentroid, near_cent_param_grid, "Nearest Centroid")        


