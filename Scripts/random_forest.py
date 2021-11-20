from get_vec_models import get_vec_models
from sklearn.ensemble import RandomForestClassifier







def run_random_forrest(model_name): 

    rf = RandomForestClassifier()





if __name__ == "__main__":

    model_dict = get_vec_models()
    for name, _ in model_dict: 
        run_random_forrest(name)

