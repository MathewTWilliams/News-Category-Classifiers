#Author: Matt Williams
#Version: 11/7/2021

#Reference: https://dataaspirant.com/nlp-text-preprocessing-techniques-implementation-python/#t-1600081660730 
#Many of the NLP related preprocessing techniques found in this file come from or are influenced by this webpage. 

#Definition for Sentence_Matrix: the sentence matrix is a 2d list that contains the words of the text the file
#is currently cleaning. Each row is a sentence and each element in the row is a word in that sentence. 

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



def tokenize_to_sents(text): 
    """Given a string of text, tokenize the text via sentences. Returns a list of
        strings where each string is a sentence from the given text. """
    return nltk.sent_tokenize(text)

def lemmatize_text(sentence_matrix):

    """Given a sentence matrix, return an updated sentence matrix, where
    each word in the original matrix was put through the WordNetLemmatizer"""

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


    updated_sent_mat = []
    lemmatizer = WordNetLemmatizer()
    for sentence in sentence_matrix:
    
        tagged_words = nltk.pos_tag(sentence)
        tagged_words = map(lambda word_tag: (word_tag[0], convert_to_wordnet_tag(word_tag[1])), tagged_words)

        lemma_words = []
        for (word, tag) in tagged_words: 

            if tag == None: 
                lemma_words.append(word)
            else: 
                lemma_words.append(lemmatizer.lemmatize(word, tag))
        
        updated_sent_mat.append(lemma_words)
    
    return updated_sent_mat

        

def stem_text(sentence_matrix):
    """Given a sentence_matrix, return a updated sentence_matrix
        where each word in the original matrix
        was put througth the SnowballStemmer(Porter2)."""
    stemmer = SnowballStemmer('english')
    updated_sent_mat = []
    for sentence in sentence_matrix:
        updated_sent = [] 
        for word in sentence:
            updated_sent.append(stemmer.stem(word))
        updated_sent_mat.append(updated_sent)
    
    return updated_sent_mat


def expand_contractions(sentences, contraction_map): 
    """Given a list of sentences and a contraction map, expand
        each contraction found in a sentence. """


    contractions_pattern = re.compile('({})'.format('|'.join(contraction_map.keys())),
                                                    flags=re.IGNORECASE|re.DOTALL)
    
    def expand_contraction(contraction):
        match = contraction.group(0)
        return contraction_map.get(match)

    expanded_sents = []
    for sentence in sentences: 
        expanded_sent = contractions_pattern.sub(expand_contraction, sentence)
        expanded_sents.append(re.sub("'", "", expanded_sent))


    return expanded_sents


def tokenize_to_words(text):
    """Given a string of text, tokenize the text and return it as a list of words."""
    return nltk.word_tokenize(text)
    
def lower_text(sentences):
    """Given a list of sentences, return the sentences where each 
        character is converted to lowercase."""
    return [sentence.lower() for sentence in sentences]

def remove_special_chars(sentences, remove_digits = False):
    """Given a list of sentences, remove all special characters from those sentences and return the
        updated sentences. Method gives the option to remove digits."""
    # used to isolate special characters that might be next to a character.
    isolate_pattern = re.compile(r'([{.(-)!}]|-)')
    special_char_pattern = re.compile(r'[^0-9a-zA-Z\s]+|\[|\]' if not remove_digits else r'[^-a-zA-Z\s]+|\[|\]')

    updated_sents = []
    for sentence in sentences: 
        sentence = isolate_pattern.sub(" \\1 ", sentence)
        sentence = special_char_pattern.sub("", sentence)
        updated_sents.append(sentence)
 
    
    return updated_sents

def remove_stop_words(sentence_matrix):
    """Given a sentence matrix, return a updated sentence matrix where
        each sentence no longer contains any stopwords (from the list
        of english stopwords from NLTK library.)"""
    stopwords = nltk.corpus.stopwords.words('english')
    updated_sent_mat = []
    for sentence in sentence_matrix:
        updated_sent = [word for word in sentence if word not in stopwords]
        updated_sent_mat.append(updated_sent)
    return updated_sent_mat

def remove_rare_words(sentence_matrix, num_to_remove = 10): 
    """Given a sentence matrix and a number of words to remove. Remove the n rarest words
        from the sentence matrix, then return the updated sentence matrix."""
    words = [word for sentence in sentence_matrix for word in sentence]
    freq_dist = FreqDist(words)
    least_common_words = freq_dist.most_common()[(num_to_remove * -1):]

    updated_sent_mat = []
    for sentence in sentence_matrix:
        update_sent = [word for word in sentence if word not in least_common_words]
        updated_sent_mat.append(update_sent)
    return updated_sent_mat
 

def remove_frequent_words(sentence_matrix, num_to_remove = 10): 
    """Given a sentence matrix and a number of words to remove. Remove the n most common words
        from the sentence matrix, then return the updated sentence matrix."""
    words = [word for sentence in sentence_matrix for word in sentence]
    freq_dist = FreqDist(words)
    most_common_words = freq_dist.most_common(num_to_remove)
    
    updated_sent_mat = []
    for sentence in sentence_matrix:
        updated_sent = [word for word in sentence if word not in most_common_words]
        updated_sent_mat.append(updated_sent)

    return updated_sent_mat

def num_to_words(sentences): 
    """Given a list of sentences, convert any number to words, then return
    the list of updated sentences."""
    updated_sents = []
    for sentence in sentences: 

        words = tokenize_to_words(sentence)

        for i in range(len(words)):
            if words[i].isdigit() or words[i].isdecimal():
                words[i] = num2words(words[i])

            updated_sents.append(" ".join(words))

    return updated_sents


def remove_single_chars(sentence_matrix):
    """Given a sentence matrix, remove any single character words from the sentence matrix. 
    Then return the updated sentence matrix."""
    update_sent_mat = []
    for sentence in sentence_matrix:
        updated_sent = [word for word in sentence if len(word) > 1]
        update_sent_mat.append(updated_sent)
    
    return update_sent_mat

def fix_spelling(sentence_matrix): 
    """Given a sentence matrix, run each word though a Spell Checker, then return 
    the updated sentence matrix."""
    checker = SpellChecker() 
    updated_sent_mat = []
    for sentence in sentence_matrix:
        updated_sent = [checker.correction(word) for word in sentence]
        updated_sent_mat.append(updated_sent)
    return updated_sent_mat




def clean_text(text, remove_digits = True, num_to_word = False, 
                rem_single_chars = True, rem_stop_words = True, 
                rem_special_chars = True, lemma = True, 
                stem = False, expand = True, lower = True, 
                num_freq_words_remove = 0, num_rare_words_remove = 0):
    """ This method is called in order to clean a given text. The method
        allows for cleaning options, however the default settings are the preferred ones.
        The method returns a sentence_matrix."""

    #make our subset of the contractions_dict from the contractions package.
    #Only keys with "’" or "'" from the original map will be present.
    contraction_map = {}
    for key, val in contractions.contractions_dict.items():
        if re.search("’|'", key):
            key = key.replace("'", "’")
            contraction_map[key] = val

    #a list of sentences
    sentences = tokenize_to_sents(text)


    if lower:
        sentences = lower_text(sentences)


    if expand:
        sentences = expand_contractions(sentences, contraction_map)

    if num_to_word and not remove_digits: 
        sentences = num_to_words(sentences)
        
    if rem_special_chars: 
        sentences = remove_special_chars(sentences, remove_digits)

    # a 2d list, each row is a sentence, each column is a word in a sentence
    sentence_matrix = []
    for sentence in sentences: 
        sentence_matrix.append(tokenize_to_words(sentence))
        

    if lemma: 
        sentence_matrix = lemmatize_text(sentence_matrix)

    if stem and not lemma: 
        sentence_matrix = stem_text(sentence_matrix)

    sentence_matrix = fix_spelling(sentence_matrix)

    if rem_stop_words: 
        sentence_matrix = remove_stop_words(sentence_matrix)

    if rem_single_chars:
        sentence_matrix = remove_single_chars(sentence_matrix)

    sentence_matrix = remove_frequent_words(sentence_matrix, num_freq_words_remove)
    sentence_matrix = remove_rare_words(sentence_matrix, num_rare_words_remove)
    
    return sentence_matrix

    
if __name__ == "__main__":
    #Test our text cleaner method.
    with open(TEST_TEXT_PATH, "r",  encoding='utf8') as file: 
        text = file.read()
        
        sentence_matrix = clean_text(text)
        updated_sents = []
        for sentence in sentence_matrix: 
            updated_sents.append(" ".join(sentence))
        
        print(" ".join(updated_sents))


        

    



