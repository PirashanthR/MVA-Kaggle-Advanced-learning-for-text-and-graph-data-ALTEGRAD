'''
preprocess -- ALTEGRAD Challenge Fall 2017 -- RATNAMOGAN Pirashanth -- SAYEM Othmane

This file contains the function that allows to do preprocessing on the raw
text read in the csv files.
'''

stopwords= []
import re
import nltk
#from nltk.corpus import stopwords
# requires nltk 3.2.1
from nltk import pos_tag
from string import punctuation
from autocorrect import spell

stopwords= ['a','the','an','is','s','usually']

def preprocess_raw_text(raw_text,stemming=False,check_spelling=False):
    global stopwords
    copy_text = raw_text
    copy_text = copy_text.lower()
    copy_text=copy_text.replace('/','or ')
    copy_text=copy_text.replace('.',' ')
    copy_text= copy_text.replace(',',' ')
    copy_text= copy_text.replace(';',' ')
    copy_text= copy_text.replace('-',' ')
    copy_text= copy_text.replace('?',' ?')
    copy_text= copy_text.replace('!',' ')
    copy_text= copy_text.replace('\"',' ')
    copy_text= copy_text.replace('(',' ')
    copy_text= copy_text.replace(')',' ')
    copy_text= copy_text.replace('PC"','computer')
    copy_text= copy_text.replace(' pc "','computer')
    copy_text= copy_text.replace('iphone"','phone')
    copy_text= copy_text.replace(r"what's",'what is')
    copy_text= copy_text.replace(r"What's",'What is')
    copy_text= copy_text.replace(r"who's",'who is')
    copy_text= copy_text.replace(r"Who's",'who is')
    copy_text= copy_text.replace(r"i'm",'i am')
    copy_text= copy_text.replace(r"\'ll",' will')
    copy_text= copy_text.replace(r"can't",' can not')
    copy_text= copy_text.replace(r"don't",' do not')
    copy_text= copy_text.replace(r"\'d",' would')
    copy_text= copy_text.replace(r"n't",' not')
    copy_text= copy_text.replace(r"e-mail",' email')
    copy_text= copy_text.replace("\'",' ')
    copy_text= copy_text.replace(r" kg ",' kilogrammes')
    copy_text= copy_text.replace(r"dual",'double')
    copy_text= copy_text.replace(r"donald trump",'trump')
    copy_text= copy_text.replace(r"hillary clinton",'clinton')
    copy_text = copy_text.replace(r" usa ", " america ")
    copy_text = copy_text.replace(r" USA ", " america ")
    copy_text = copy_text.replace(r" us ", " america ")
    copy_text = copy_text.replace(r" uk ", " england ")
    copy_text = copy_text.replace(r" UK ", " england ")
    copy_text = copy_text.replace(r" one ", " 1 ")
    copy_text = copy_text.replace(r" two ", " 2 ")
    copy_text = copy_text.replace(r" three ", " 3 ")
    copy_text = copy_text.replace(r" four ", " 4 ")
    copy_text = copy_text.replace(r" five ", " 5 ")
    copy_text = copy_text.replace(r" six ", " 6 ")
    copy_text = copy_text.replace(r" seven ", " 7 ")
    copy_text = copy_text.replace(r" eight ", " 8 ")
    copy_text = copy_text.replace(r" nine ", " 9 ")
    copy_text = copy_text.replace(r" iit ", " indian institutes of technology ")
    copy_text = copy_text.replace(r" iim ", " indian institutes of management ")
    copy_text = copy_text.replace(r" gr ", " general relativity ")
    copy_text = copy_text.replace(r" qm ", " quantum mechanics ")
    copy_text = copy_text.replace(r" ms ", " master ")
    copy_text = copy_text.replace(r" instagram ", " photos website ")
    copy_text = copy_text.replace(r" quora ", " questions website ")
    copy_text = copy_text.replace(r" six pack ", " six packs ")
    copy_text = copy_text.replace(r" 40s ", " 40 year old ")
    copy_text = copy_text.replace(r" java se ", " java standard edition ")
    
    copy_text = re.sub(' +',' ',copy_text)
    
    copy_text= copy_text.strip() # strip extra white space
    
    split_text= copy_text.split()
    stemmer = nltk.stem.SnowballStemmer('english')
    if stemming:
        tokens_stemmed = list()
        for token in split_text:
            tokens_stemmed.append(stemmer.stem(token))
        split_text = tokens_stemmed
    
    if check_spelling:
        correct_tokens = list()
        for token in split_text:
            correct_tokens.append(spell(token))
        split_text= correct_tokens
    
    return ' '.join([t for t in split_text if t not in stopwords])
