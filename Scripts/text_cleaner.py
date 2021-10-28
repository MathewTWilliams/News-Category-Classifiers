#Author: Matt Williams
#Version: 10/27/2021

#Reference: https://dataaspirant.com/nlp-text-preprocessing-techniques-implementation-python/#t-1600081660730 
#Many of the NLP related preprocessing techniques found here come from or are influenced by this webpage. 


from bs4 import BeautifulSoup
from contractions import contractions_dict
import re
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
import nltk
from num2words import num2words
from constants import TEST_TEXT_PATH


def lemmatize_text(words):
    """Given a list of words, return a list of words 
    where each word is ran through the WordNetLemmatizer"""
    tagged_words = nltk.pos_tag(words)
    lemma_words = []
    lemmatizer = WordNetLemmatizer()
    for (word, tag) in tagged_words: 
        lemma_words.append(lemmatizer.lemmatize(word, tag))

    return lemma_words
        

def stem_text(words):
    """Given a list of words, return a list of words
    where each words is ran througth the SnowballStemmer(Porter2)."""
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(word) for word in words]


def expand_contractions(text): 
    contractions_pattern = re.compile('({})'.format('|'.join(contractions_dict.keys())),
                                                    flags=re.IGNORECASE|re.DOTALL)
    
    def expand_contraction(contraction):
        match = contraction.group(0)
        return contractions_dict.get(match)

    expanded_text = contractions_pattern.sub(expand_contraction, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def tokenize(text):
    return nltk.word_tokenize(text)
    
def lower_text(text):
    return text.lower()

def remove_special_chars(text, remove_digits = False):
    # used to isolate special characters that might be next to a character.
    special_char_pattern = re.compile(r'({.(-)!})')
    text = special_char_pattern.sub(" \\1 ", text)

    # remove the special characters
    pattern = r'[^0-9a-zA-Z\s]+|\[|\]' if not remove_digits else r'[^-a-zA-Z\s]+|\[|\]'
    text = re.sub(pattern,'', text)
    return text

def remove_stop_words(words):
    stopwords = nltk.corpus.stopwords.words('english')
    return [word for word in words if word not in stopwords]

def remove_rare_words(words, num_to_remove = 10): 
    freq_dist = FreqDist(words)
    least_common_words = freq_dist.most_common()[(num_to_remove * -1):]
    return [word for word in words if word not in least_common_words]
 

def remove_frequent_words(words, num_to_remove = 10): 
    freq_dist = FreqDist(words)
    most_common_words = freq_dist.most_common(num_to_remove)
    return [word for word in words if word not in most_common_words]





#in HuffPost articles, any div with the class ="primary-cli cli cli-text"
#contains a portion of the article body. 
def clean_html(html): 

    soup = BeautifulSoup(html, "html.parser")
    content = soup.find_all("div", 
                        {"class": "primary-cli cli cli-text"})

    return "".join([child.get_text() for child in content])


def num_to_words(words): 
    for i in range(len(words)):
        if words[i].isdigit() or words[i].isdecimal():
            words[i] = num2words(words[i])
    
    updated_text = " ".join(words)
    return tokenize(updated_text)


def remove_single_chars(words):
    return [word for word in words if len(word) > 1]

def get_and_clean_text(text, remove_digits = False, num_to_word = True, 
                            rem_single_chars = True, remove_stop_words = True, 
                            remove_special_chars = True, lemmatize_text = True, 
                            stem_text = False, expand_contractions = True, 
                            tokenize = True, lower_text = True, 
                            num_freq_words_remove = 0, num_infreq_words_remove = 0):


    if lower_text:
        text = lower_text(text)
    
    if expand_contractions:
        text = expand_contractions(text)
    
    if remove_special_chars: 
        text = remove_special_chars(remove_digits)

    if  
    

if __name__ == "__main__":
    with open(TEST_TEXT_PATH, "r",  encoding='utf8') as file: 
        text = file.read()
       


