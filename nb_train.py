import pandas as pd
import string
import re
import nltk
import random
import pickle
from sklearn.model_selection import train_test_split
from nltk.corpus import stopwords
from zipfile import ZipFile

with ZipFile('imdb ratings.zip') as zf:
    f = zf.open('movie.csv')
    
df = pd.read_csv(f)
smalldf = df[:2000]
smalldf = smalldf.drop_duplicates(ignore_index = True)
smalldf.loc[smalldf['label'] == 0, 'label'] = 'neg'
smalldf.loc[smalldf['label'] == 1, 'label'] = 'pos'

def word_processor(text):  
    text = "".join([i.lower() for i in text.replace('<br', '') if i not in string.punctuation])
    text = re.sub('\s+',' ', text)
    text = ' '.join([word for word in text.split() if word not in stopwords.words('english')])
    return text

def rev_sent(df):  
    '''pair processed review strings with their corresponding class labels in tuples, 
    return the tuples in a randomly shuffled list'''
    
    tup_list = []
    for i in range(len(df)):
        new_string = word_processor(df.iloc[i]['text'])
        sentiment = df.iloc[i]['label']
        tup_list.append((new_string.split(' '), sentiment))
    
    random.shuffle(tup_list)
    
    return tup_list 

def commwords_extractor(tuplist, x):
    '''get the x number of most common words out of the strings in tup_list, return the words in a list'''
    
    review_words = [t[0] for t in tuplist]
    words_flat = [word for rev in review_words for word in rev] 
    all_words = nltk.FreqDist(w.lower() for w in words_flat) 
    comm_words = list(all_words)[:x]
    
    return comm_words

def review_features(review):
    '''decompose a review into unique words, returns a dict mapping features to booleans if the feature is in a given review'''
    
    words = set(review) 
    features = {}
    for w in commwords:
        features[f'contains({w})'] = (w in words)
        
    return features

def train_model(data):
    '''Train the model, return the model'''
    
    classifier = nltk.NaiveBayesClassifier.train(data)   
    
    return classifier


#Pair review to corresponding sentiment
review_to_sentiment = rev_sent(smalldf)

#Extract 2000 most common words
commwords = commwords_extractor(review_to_sentiment, 2000)

def main():
    
    featuresets = [(review_features(rev), label) for (rev, label) in review_to_sentiment]
    train_set, test_set = featuresets[:1500], featuresets[1500:]
    
    model_accuracy = nltk.classify.accuracy(train_model(train_set), test_set)
    print('NB model accuracy:', model_accuracy)
    print(train_model(train_set).show_most_informative_features(5))
    
    #Save model
    filename = 'nb_classifier'
    outfile = open(filename,'wb')
    pickle.dump(train_model(train_set), outfile)
    outfile.close()
    
if __name__ == '__main__':
    main()