#Author: Matt Williams
#Version: 10/31/2021

#Reference: https://dataaspirant.com/nlp-text-preprocessing-techniques-implementation-python/#t-1600081660730 
#Many of the NLP related preprocessing techniques found here come from or are influenced by this webpage. 


import contractions
import re
from nltk.probability import FreqDist
from nltk.stem.wordnet import WordNetLemmatizer
from nltk.stem import SnowballStemmer
from nltk.corpus import wordnet
import nltk
from num2words import num2words
from constants import TEST_TEXT_PATH
from spellchecker import SpellChecker




def lemmatize_text(words):

    """Given a list of words, return a list of words 
    where each word is ran through the WordNetLemmatizer"""

    #referece: https://gaurav5430.medium.com/using-nltk-for-lemmatizing-sentences-c1bfff963258 
    def convert_to_wordnet_tag(nltk_tag): 
        if nltk_tag.startswith('J'): 
            return wordnet.ADJ
        elif nltk_tag.startswith("V"): 
            return wordnet.VERB
        elif nltk_tag.startswith("N"): 
            return wordnet.NOUN
        elif nltk_tag.startswith("R"): 
            return wordnet.ADV
        
        return None


    tagged_words = nltk.pos_tag(words)
    tagged_words = map(lambda word_tag: (word_tag[0], convert_to_wordnet_tag(word_tag[1])), tagged_words)




    lemma_words = []
    lemmatizer = WordNetLemmatizer()
    for (word, tag) in tagged_words: 

        if tag == None: 
            lemma_words.append(word)
        else: 
            lemma_words.append(lemmatizer.lemmatize(word, tag))

    return lemma_words
        

def stem_text(words):
    """Given a list of words, return a list of words
    where each words is ran througth the SnowballStemmer(Porter2)."""
    stemmer = SnowballStemmer('english')
    return [stemmer.stem(word) for word in words]


def expand_contractions(text, contraction_map): 

    


    contractions_pattern = re.compile('({})'.format('|'.join(contraction_map.keys())),
                                                    flags=re.IGNORECASE|re.DOTALL)
    
    def expand_contraction(contraction):
        match = contraction.group(0)
        return contraction_map.get(match)

    expanded_text = contractions_pattern.sub(expand_contraction, text)
    expanded_text = re.sub("'", "", expanded_text)
    return expanded_text


def tokenize_text(text):
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


def num_to_words(text): 

    words = tokenize_text(text)

    for i in range(len(words)):
        if words[i].isdigit() or words[i].isdecimal():
            words[i] = num2words(words[i])

    return " ".join(words)


def remove_single_chars(words):
    return [word for word in words if len(word) > 1]

def fix_spelling(words): 
    checker = SpellChecker() 
    return [checker.correction(word) for word in words]




def clean_text(text, remove_digits = True, num_to_word = False, 
                rem_single_chars = True, rem_stop_words = True, 
                rem_special_chars = True, lemma = True, 
                stem = False, expand = True, 
                tokenize = True, lower = True, 
                num_freq_words_remove = 0, num_rare_words_remove = 0):
    """The default settings currenlty are the preferred settings."""

    #make our subset of the contractions_dict from the contractions package.
    #Only keys with "’" or "'" from the original map will be present.
    contraction_map = {}
    for key, val in contractions.contractions_dict.items():
        if re.search("’|'", key):
            key = key.replace("'", "’")
            contraction_map[key] = val



    if lower:
        text = lower_text(text)


    if expand:
        text = expand_contractions(text, contraction_map)

    if num_to_word and not remove_digits: 
        text = num_to_words(text)
        
    if rem_special_chars: 
        text = remove_special_chars(text, remove_digits)

    if tokenize: 
        words = tokenize_text(text)
    
    if lemma: 
        words = lemmatize_text(words)

    if stem and not lemma: 
        words = stem_text(words)

    words = fix_spelling(words)


    if rem_stop_words: 
        words = remove_stop_words(words)

    if rem_single_chars:
        words = remove_single_chars(words)

    words = remove_frequent_words(words, num_freq_words_remove)
    words = remove_rare_words(words, num_rare_words_remove)
    
    return " ".join(words)

    
if __name__ == "__main__":
    with open(TEST_TEXT_PATH, "r",  encoding='utf8') as file: 
        text = file.read()
        
        print(clean_text(text))
    



