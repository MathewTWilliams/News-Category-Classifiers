#Author: Matt Williams
#Version: 10/19/2021


from bs4 import BeautifulSoup
import contractions
import re
from nltk.tokenize.treebank import TreebankWordTokenizer



#html_stripping - got it
#tokenization - got it 
#contraction expansion
#accented_char_removal
#text_lower_case
#text_stemming
#text_lemmatization
#Punctuation/Special Character removal
#remove digits
#stopword_removal
#Removal of Punctuation
#spelling correction
#number to words
#Removing Frequent Words
#removing rare words
#removed extra white space
#Remove Single characters


def tokenize_text(text):
    tokenizer = TreebankWordTokenizer()
    return tokenizer.tokenize(text)



#in HuffPost articles, any div with the class ="primary-cli cli cli-text"
#contains a portion of the article body. 
def clean_html(html): 

    soup = BeautifulSoup(html, "html.parser")
    content = soup.find_all("div", 
                        {"class": "primary-cli cli cli-text"})

    text = ""
    for child in content: 
        text += child.get_text()

    return text



