#Author: Matt Williams
#Version: 12/02/2021


import gensim.downloader as api
from utils import get_model_path


def download_models():
    '''Download our selected pre-trained models and save them for later use.'''
    download_names = ["word2vec-google-news-300", 
            "glove-wiki-gigaword-300", 
            "fasttext-wiki-news-subwords-300"]

    #load our pretrained models save their word vectors
    for name in download_names:
        model = api.load(name)
        model_type = name.split('-')[0]
        model.save(get_model_path(model_type + '.model'))





if __name__ == "__main__":
    download_models()