import os
import gensim
from constants import W2V_DIR_PATH, get_w2v_path


def get_w2v_models(): 
    """Returns a list of word2vec gensim models found in the models folder."""
    models = []

    if not os.path.exists(W2V_DIR_PATH):
        return None
    

    for file in os.listdir(W2V_DIR_PATH):
        if file.split(".")[-1] == "model":
            w2v = gensim.models.Word2Vec.load(get_w2v_path(file))
            models.append(w2v)

    return models

if __name__ == "__main__":
    print(len(get_w2v_models()))

