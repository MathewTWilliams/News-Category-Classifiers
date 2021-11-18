from pickle import load
from save_load_json import load_json
from constants import TRAIN_SET_PATH, get_w2v_path
from gensim.models import Word2Vec, FastText



def make_vec_model(sentences, sg, vector_size = 100, window = 5, min_count = 5, 
                        is_fast_text = False , alpha = 0.025, epochs = 5):
    """sg = 1 mean Skip Gram. 0 means CBOW. All other 
    parameters are optional and are set to the default values used
    by gensim. Both models will be using negative sampling instead of
    hierarchical softmax"""

    if is_fast_text:
        vector = FastText(sentences=sentences, sg = sg, vector_size= vector_size, 
                            window= window, min_count=min_count, hs=0, alpha=alpha, 
                            epochs = epochs)
        name = "FT_"
        name += "SKPGRM" if sg == 1 else "CBOW"
        vector.wv.save(get_w2v_path(name + ".model"))

    else:
        vector = Word2Vec(sentences=sentences, sg = sg, vector_size= vector_size, 
                            window= window, min_count=min_count, hs=0, alpha=alpha, 
                            epochs = epochs)

        name = "SKPGRM" if sg == 1 else "CBOW"
        vector.wv.save(get_w2v_path(name + ".model"))


if __name__ == "__main__": 
    train_text_dict = load_json(TRAIN_SET_PATH)
    sentences = [sentence \
                for _, article_list in train_text_dict.items()  \
                for article in article_list \
                for sentence in article]
    make_vec_model(sentences = sentences, sg=0, is_fast_text=False)
    make_vec_model(sentences = sentences, sg=1, is_fast_text=False)
    make_vec_model(sentences = sentences, sg=0, is_fast_text=True)
    make_vec_model(sentences = sentences, sg=1, is_fast_text=True)




