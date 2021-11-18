import os
from gensim.models import KeyedVectors
from constants import MODEL_DIR_PATH, get_w2v_path


def get_vec_models(): 
    """Returns a list of word2vec gensim models found in the models folder."""
    models = {}

    if not os.path.exists(MODEL_DIR_PATH):
        return None
    

    for file in os.listdir(MODEL_DIR_PATH):
        name_split = file.split('.')
        if name_split[-1] == "model":
            vec = KeyedVectors.load(get_w2v_path(file))
            models[name_split[0]] = vec
       
    return models

if __name__ == "__main__":
    models = get_vec_models()
    print(len(models.items()))
