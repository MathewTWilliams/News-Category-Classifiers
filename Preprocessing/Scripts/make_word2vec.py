import gensim
from save_load_json import load_json
from constants import CLEANED_TEXT_PATH, get_w2v_path
from datetime import datetime



def save_word2vec(vector, name_prefix):
    name = name_prefix + "_" + datetime.now().strftime("%Y%m%d-%H%M%S")
    vector.save(get_w2v_path(name + ".model"))
    
    return name

def save_word2vec_details(file_name, details): 
    with open(get_w2v_path(file_name), "w", encoding="utf-8") as file: 
        file.write(details)


def make_word2vec(sg, vector_size = 100, window = 5, min_count = 5, 
                        hs = 0, alpha = 0.025, epochs = 5, 
                        min_alpha = 0.0001, sample = 0.001):
    """sg = 1 mean Skip Gram. 0 means CBOW. All other 
    parameters are optional and are set to the default values used
    by gensim"""

    clean_text_dict = load_json(CLEANED_TEXT_PATH)
    sentences = [sentence \
                for _, article_list in clean_text_dict.items()  \
                for article in article_list \
                for sentence in article]


    vector = gensim.models.Word2Vec(sentences=sentences, sg = sg, vector_size= vector_size, 
                                    window= window, min_count=min_count, hs=hs, alpha=alpha, 
                                    epochs = epochs, min_alpha=min_alpha, sample=sample)
    
    prefix = "SKPGRM" if sg == 1 else "CBOW"
    details_file_name = save_word2vec(vector, prefix) + "_details.txt"

    details = []
    details.append("Is Skip Gram = " + str(sg == 1))
    details.append("Vector Size: " + str(vector_size))
    details.append("Window Size: " + str(window))
    details.append("Min Word Count: " + str(min_count))
    details.append("Using Hierarichal Softmax: " + str(hs == 1))    
    details.append("Alpha: " + str(alpha))
    details.append("Epochs: " + str(epochs))
    details.append("Minimum alpha: " +  str(min_alpha))
    details.append("Down Sample Threshold: " +  str(sample))

    save_word2vec_details(details_file_name, "\n".join(details))


if __name__ == "__main__": 
    make_word2vec(sg = 1)
    make_word2vec(sg = 0)



