from sklearn.cluster import KMeans
import numpy as np
from get_article_vectors import get_training_info, get_test_info
from get_vec_models import get_model_names

#Parameters for Random Search Cross Validation
#Used to help condense our search for the best hyperparameters to use
inits = ["k-means++", "random"]
algorithms = ["full", "elkan"]
max_iters = [int(x) for x in np.linspace(start = 100, stop = 500, num = 5)]


#Parameters for regular Cross Validation
#These parameters are chosen after running Random Search Cross Validation


def run_k_means(model_name, init = "k-means++", algorithm = "elkan", max_iter = 300):
    #we have 10 categories so the number of clusters will be 10
    training_data, _ = get_training_info(model_name)
    test_data, test_labels = get_test_info(model_name)


if __name__ == "__main__": 
    print()