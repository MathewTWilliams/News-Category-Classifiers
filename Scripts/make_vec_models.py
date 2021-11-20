from save_load_json import load_json
from constants import CLEANED_TEXT_PATH, get_w2v_path
from gensim.models import Word2Vec
from gensim.models.callbacks import CallbackAny2Vec



class EpochCallback(CallbackAny2Vec):

    def __init__(self):
        self.epoch = 1
    
    def on_epoch_end(self, model):
       loss = model.get_lastest_training_loss()
       print("Training Loss on Epoch", self.epoch, ":", loss)
       self.epoch += 1

def make_vec_model(sentences, sg, vector_size = 100, window = 5, 
                         alpha = 0.025, epochs = 5, sample = 0.001):
    """sg = 1 mean Skip Gram. 0 means CBOW. All other 
    parameters are optional and are set to the default values used
    by gensim. Both models will be using negative sampling instead of
    hierarchical softmax"""

   
    vector = Word2Vec(sentences=sentences, sg = sg, vector_size= vector_size, 
                        window= window, min_count=1, hs=0, alpha=alpha, 
                        epochs = epochs, sample=sample, callbacks = [EpochCallback()])

    name = "SKPGRM" if sg == 1 else "CBOW"
    vector.wv.save(get_w2v_path(name + ".model"))


if __name__ == "__main__": 
    cleaned_text_dict = load_json(CLEANED_TEXT_PATH)
    sentences = [sentence \
                for _, article_list in cleaned_text_dict.items()  \
                for article in article_list \
                for sentence in article]

    make_vec_model(sentences = sentences, sg=0, vector_size = 250, epochs = 10)
    make_vec_model(sentences = sentences, sg=1, window = 10, vector_size = 250, epochs = 10)





