from mlp import run_mlp
from svm import run_svm
from random_forest import run_random_forest
from near_radius import run_near_radius
from near_centroid import run_near_centroid
from naive_bayes import run_naive_bayes
from logistic_regression import run_logistic_regression
from knn import run_knn
from grad_boost import run_grad_boost
from bagging import run_bagging
from ada_boost import run_ada


from get_vec_models import get_vec_model_names

if __name__ == "__main__": 

    for vec_model in get_vec_model_names(): 
        run_mlp(vec_model)
        run_svm(vec_model)
        run_random_forest(vec_model)
        #run_near_radius(vec_model)
        run_near_centroid(vec_model)
        run_naive_bayes(vec_model)
        run_logistic_regression(vec_model)
        run_knn(vec_model)
        #run_grad_boost(vec_model)
        run_bagging(vec_model)
        run_ada(vec_model)