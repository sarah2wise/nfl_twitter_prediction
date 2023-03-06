# Built from code from: https://www.pluralsight.com/guides/building-a-twitter-sentiment-analysis-in-python

import pandas as pd
import numpy as np
import re
import string
from nltk.corpus import stopwords
from nltk.tokenize import wordpunct_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from nltk.stem import PorterStemmer
from nltk.stem import LancasterStemmer
from nltk.stem.util import prefix_replace
from nltk.stem.util import suffix_replace
from nltk.stem import WordNetLemmatizer
import nltk
# run first time
# nltk.download('stopwords')
# nltk.download('punkt')
# nltk.download('wordnet')
# nltk.download('omw-1.4')

# Read in file
df = pd.read_excel("Sample Tweets.xlsx", sheet_name = 'Ratings').drop(columns = ['Unnamed: 0'])

# count CAPITAL words, excluding I and A - does include positions and acronyms though
df['capitals'] = [len(re.findall(r'\b[A-Z]+\b(?<!I|A)', tweet)) for tweet in df['text']]
# count !!!S
df['!s'] = [tweet.count("!") for tweet in df['text']]

df[['text','capitals']]
df[['text','!s']]


# Making stop word list
init_stop_words = set(stopwords.words('english'))
stop_words = set()
# add stop words to list without apostrophes
for stop_word in init_stop_words:
    if "'" in stop_word:
        ind = stop_word.find("'")
        stop_words.add(stop_word[0:ind] + stop_word[ind+1:])
    else:
        stop_words.add(stop_word)

stop_words.add('one') # substitute for a
stop_words.add('hes') # missing
stop_words.add('got') # got to
stop_words.add('going') # going to
stop_words.add('ta') # slang (got + ta = gotta)
stop_words.add('gon') # slang (gon + na = gonna)
stop_words.add('na') # slang (gon + na = gonna)
stop_words.add('aint') # slang
stop_words.add('theyre') # missing
stop_words.add('da') # da bears
stop_words.add('st') # 21st
stop_words.add('nd') # 22nd
stop_words.add('rd') # 23rd
stop_words.add('th') # 24-30th

stop_words.remove('no') # remove no from stop words because NO = New Orleans 
stop_words.remove('not') # remove not from stop words because shows strong negation
neg_list = ['aint', 'arent', 'couldnt', 'didnt', 'doesnt', 'dont', 
            'hadnt', 'havent', 'hasnt', 'isnt', 'mightnt', 'mustnt', 
            'neednt', 'shant', 'shouldnt', 'wasnt', 'werent', 'wont','wouldnt'] # negation words - not stop words
for neg in neg_list:
    stop_words.remove(neg)
context_list = ['above', 'below', 'under', 'down', 'up', 'against', 'for', 'very'] # not stop words - may need for context
for con in context_list:
    stop_words.remove(con)

# abreviation dict, add spaces to account for words alone, otherwise will change every letter
abr_dict = {
    # football specific terms
    ' td' : 'touchdown', # td or tds
    ' int ' : ' interception ',
    ' yd' : 'yard', # yd or yds
    ' pi ' : ' pass interference ',
    ' dl ' : ' defensive line ',
    ' ol ' : ' offensive line ',
    ' o ' : ' offense ',
    ' offence ' : ' offense ', # misspelling from the train set
    ' d ' : ' defense ',
    ' rec ' : ' receptions', # unsure about this one...
    ' pup ' : ' physically unable to perform ',
    ' opi ' : ' offensize pass interference ',
    ' dpi ' : ' defensive pass interference ',
    ' pi ' : ' pass interference ',
    ' yac ' : ' yards after catch ',
    ' mvp ' : ' most valuable player ',
    ' st ' : ' special teams ',
    # positions
    ' hc ' : ' head coach ', # in training set
    ' oc ' : 'offensize coordinator',
    ' dc ' : 'defensive coordinator',
    ' qb' : ' quarterback ', # in training set
    ' te ' : ' tight end ', # in training set
    ' k ' : ' kicker ', # in training set
    ' rb ' : ' running back ', # in training set
    ' wr ' : ' wide receiver ', # in training set
    ' db ' : ' defensive back ',
    ' fb ' : ' full back ',
    ' hb ' : ' half back ',
    ' cb ' : ' corner back ', # in training set
    ' de ' : ' defensive end ',
    ' dt ' : ' defensive tackle ', # in training set
    ' lb ' : ' linebacker ', # in training set
    ' rt ' : ' right tackle ', # in training set
    ' lt ' : ' left tackle ',
    # shorthand
    ' w ' : ' win ',
    ' dub ' : ' win ',
    ' l ' : ' lose ',
    ' loss ' : ' lose ', # get same form of word when lemmattized
    ' bc ' : ' because ',
    # abr slang
    'wtf' : 'what the fuck',
    'lol' : 'laugh out loud',
    'tbh' : 'to be honest',
    'tbd' : 'to be determined',
    'hmu' : 'hit me up',
    ' bs ' : ' bull shit ',
    'vs' : ' versus ', # ex. NOvsTB
    'mnf' : 'monday night football',
    # numbers
    ' two ' : ' 2 ',
    ' three ' : ' 3 ',
    ' four ' : ' 4 ',
    ' five ' : ' 5 ',
    ' six ' : ' 6 ',
    ' seven ' : ' 7 ',
    ' eight ' : ' 8 ',
    ' nine ' : ' 9 ',
    ' ten ' : ' 10 ',
    ' 1st' : ' first', # 1st down, 1st game, etc.
    ' 2nd' : ' second',
    ' 3rd' : ' third',
    ' 4th' : ' fourth',
    # team specific abbreviations
    ' az ' : ' arizona ',
    ' cards ' : ' cardinals ',
    ' falcs ' : ' falcons ',
    ' cincy ' : ' cincinnati ',
    ' gb ' : ' green bay ',
    ' jags ' : ' jaguars ',
    ' jag ' : ' jaguars ',
    ' dolphs ' : ' dolphins ',
    ' fins ' : ' dolphins ',
    ' fin ' : ' dolphins ',
    ' vike ' : ' vikings ',
    ' vikes ' : ' vikings ',
    ' pat ' : ' patriots ',
    ' pats ' : ' patriots ',
    ' gmen ' : ' giants ',
    ' stillers ' : ' steelers ',
    ' bucs ' : ' buccaneers ',
    ' wft ' : ' washington football team '
    }
# translating neg_list to not for simplicity
for neg in neg_list:
    abr_dict[neg] = 'not'
    

def preprocess_tweet_text(tweet):
    # Convert all words to lowercase
    tweet = tweet.lower()
    # Remove urls
    tweet = re.sub(r"http\S+|www\S+|https\S+", '', tweet, flags=re.MULTILINE)
    # Remove user @ references and '#' from tweet
    tweet = re.sub(r'\@\w+|\#','', tweet)
    
    # Shortens multiple vowels ex. Maaaan
    tweet = re.sub(r'a{2,100}','a', tweet) # 2 or more consecutive a's
    # Shortens multiple vowels ex. Meeeee
    tweet = re.sub(r'e{3,100}','e', tweet) # 3 or more consecutive e's
    # Shortens multiple vowels ex. Gooooo
    tweet = re.sub(r'o{3,100}','o', tweet) # 3 or more consecutive o's
    # Shortens multiple vowels ex. Hiiiigh
    tweet = re.sub(r'i{2,100}','i', tweet) # 2 or more consecutive i's
    # Shortens multiple vowels ex. Truuuue
    tweet = re.sub(r'u{2,100}','u', tweet) # 2 or more consecutive u's
    # Shortens multiple vowels ex. Whyyyyy
    tweet = re.sub(r'y{2,100}','y', tweet) # 2 or more consecutive y's
    
    consonants = ['b','c','d','f','g','h','j','k','l','m','n','p','q','r','s','t','v','w','x','z']
    # Shortens multiple constonants ex. Baddddd Mannnnn
    for con in consonants:
        r_str = re.compile(con + "{3,100}")  # 3 or more consecutive consonants
        tweet = re.sub(r_str, con, tweet)
    
    # Substitutiong #-# for good/bad record (some are actually game scores though)
    new_tweet = ""
    for t in tweet.split():
        match = re.match(r'\b\d-\d.*\b', t)
        if type(match) != type(None):
            record = match[0].split('.')
            record = record[0].split('‚')
            record = record[0].split('-')
            if int(record[0]) < int(record[1]):
                new_tweet += 'bad record '
            else:
                new_tweet += 'good record '
        else:
            new_tweet += t + " "
    
    # Replace abreviations that require punctuations
    new_tweet = new_tweet.replace('w/', 'with ')
    new_tweet = new_tweet.replace('and/or', 'and or')
    new_tweet = new_tweet.replace(' d.c. ', ' dc ')

    
    # Remove punctuations
    new_tweet = re.sub(r"[!\"#\$%&\'\(\)\*\+,-\./:;<=>\?@\[\\\]\^_`{\|}~]",' ', new_tweet)
    
    # Replace abreviations that require numbers
    new_tweet = new_tweet.replace(' s ', ' safety ') # as in S and not 80s, 90s, etc.
    new_tweet = new_tweet.replace(' b4 ', ' before ')
    
    # Remove all numbers
    new_tweet = re.sub(r'\d', '', new_tweet)
    
    # Replace abreviations with punctuation and numbers gone
    for old, new in abr_dict.items():
        new_tweet = new_tweet.replace(old, new)

    tweet_tokens = wordpunct_tokenize(new_tweet) # separates words and punctuation and spellchecks (sometimes)
   
    # Remove emojis and weird (non-english alphabet) characters
    demoji = [w.encode('ascii', 'ignore').decode('ascii') for w in tweet_tokens]
    # Remove stop words
    filtered_words = [w for w in demoji if not w in stop_words]

    # stemm
    ps = PorterStemmer() # removes suffixes - makes it look very strange and unreadable, including proper nouns
    # ls = LancasterStemmer() # more aggressive suffix removal
    stemmed_words = [ps.stem(w) for w in filtered_words]
    # stemmed_words = [ls.stem(w) for w in filtered_words]
    
    # lemmatize
    lemmatizer = WordNetLemmatizer()
    lemma_words = [lemmatizer.lemmatize(w, pos='v') for w in stemmed_words] # changes verbs to same tense
    
    # Remove single letters left (d and c from d.c., s from 80s, etc.)
    final_words = [w for w in lemma_words if len(w) > 1]
    # join words again
    final_tweet = " ".join(final_words)
    # Remove extra whitespace
    # final_tweet = " ".join(final_tweet.split())

    
    return final_tweet

# Preprocess data
df.text = df['text'].apply(preprocess_tweet_text)

df.rename(columns = {'Sentiment Rating' : 'sentiment'}, inplace = True)
df.to_csv("preprocessed_tweets.csv")
