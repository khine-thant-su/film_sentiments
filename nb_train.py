import pandas as pd
import string
import re
import nltk
import random
import pickle
from nltk.corpus import stopwords
from zipfile import ZipFile

DATA = 'movie.csv'
FEATURE_0 = 'text'
LABEL = 'label'
LABEL_0 = 0
LABEL_1 = 1
NEGATIVE_SENTIMENT = 'neg'
POSITIVE_SENTIMENT = 'pos'
DATA_LENGTH = 2000
STOPWORD_TYPE = 'english'
TRAIN_DATA_LENGTH = 1500
NUM_OF_FEATURES = 5
FILE_NAME = 'nb_classifier'

class NaiveBayesClassifier:
    def __init__(self, classifier): 
        self.classifier = classifier 
    
    @classmethod
    def from_pickled(cls, path):
        '''Unpickles the file given by path, returns an instance of the class initialized with the classifier object.'''
        
        classifier = pickle.load(open(path,'rb'))
        return cls(classifier)
    
    def test(self, test_data):
        '''Evaluates the model on test data, and returns the accuracy.'''
        
        return nltk.classify.accuracy(classifier, test_data)
    
    def predict(self, str_input):
        '''Takes in a string, splits it, and returns a prediction for the sentiment of that string.'''
        
        words = str_input.split(' ')
        return classifier.classify(find_features(words, common_words))  # TODO: Will common_words defined in main() be visible to predict()?
        
    
    def store(self, path):
        '''Accepts a file path and stores the model at that file path using the pickle library.'''
        
        with open(path,'wb') as f:
            pickle.dump(classifier, f)
    

def clean(text):
    '''Converts the input text into lower case, removes <br> tags, punctuation, whitespace, and stopwords.
    Returns a list of words that remain.
    
        Args:
        text(str): input text'''
    
    text = ''.join([w.lower() for w in text.replace('<br', '') if w not in string.punctuation])
    text = re.sub('\s+',' ', text)
    text = [word for word in text.split() if word not in stopwords.words(STOPWORD_TYPE)]
    return text

def make_text_label_tuples(df): 
    '''Make tuples that pair processed strings with their corresponding labels. Returns the tuples in a list.
    
        Args:
        df: dataframe'''
    
    tup_list = []
    for i in range(len(df)):
        word_list = clean(df.iloc[i][FEATURE_0])
        sentiment = df.iloc[i][LABEL]
        tup_list.append((word_list, sentiment))  
    
    return tup_list 

def extract_common_words(data, count):
    '''Selects the specified number of most common words out of the strings in data, returns the words in a list.
    
        Args:
        data(list): a list of words
        count(int): the number of most common words to be extracted.'''
       
    word_frequency = nltk.FreqDist(w.lower() for w in data) 
    common_words = list(word_frequency)[:count]
    
    return common_words

def find_features(word_list, feature_words):
    '''Select unique words from a list of words, returns a dictionary mapping features to booleans.
 
        Args:
        word_list(list): a list of words split from a sentence
        feature_words(list): a list of common words
        
        Returns:
        features(dict): a dictionary with each word in feature_words as keys and booleans as values 
        (T if the word appears in word_list, F if otherwise).'''
    
    words = set(word_list)
    features = {}
    for w in feature_words:  
        features[f'contains({w})'] = (w in words)
        
    return features

def train_model(data):
    '''Trains the model, returns the model.
    
        Args:
        data(list): a list of tuples where the first element of each tuple is a dict of features, the second element is the sentiment.'''
    
    classifier = nltk.NaiveBayesClassifier.train(data)  
    
    return classifier


def main():
    
    with ZipFile('imdb ratings.zip') as zf:
        f = zf.open(DATA)
    
    df = pd.read_csv(f)
    smalldf = df[:DATA_LENGTH]
    smalldf = smalldf.drop_duplicates(ignore_index = True)
    smalldf.loc[smalldf[LABEL] == LABEL_0, LABEL] = NEGATIVE_SENTIMENT
    smalldf.loc[smalldf[LABEL] == LABEL_1, LABEL] = POSITIVE_SENTIMENT
    
    
    # Pair each string to its corresponding sentiment.
    str_to_sentiment = make_text_label_tuples(smalldf)  
    
    random.shuffle(str_to_sentiment) # randomly shuffle the list of tuples to randomize the contents of train and test data.
    
    nested_words = [t[0] for t in str_to_sentiment] 
    
    words_flatlist = [word for wl in nested_words for word in wl]
    
    common_words = extract_common_words(words_flatlist, 2000)
    
    # TODO: should I only find features from the train data? Am I overfitting by also finding features from the test set?
    
    featuresets = [(find_features(word_list, common_words), label) for (word_list, label) in str_to_sentiment] 
    
    train_set, test_set = featuresets[:TRAIN_DATA_LENGTH], featuresets[TRAIN_DATA_LENGTH:]
    
    model_accuracy = nltk.classify.accuracy(train_model(train_set), test_set)  
    
    print('NB model accuracy:', model_accuracy)
    print(train_model(train_set).show_most_informative_features(NUM_OF_FEATURES))
    
    # Save model.
    with open(FILE_NAME,'wb') as f:
        pickle.dump(train_model(train_set), f)

    
    model = NaiveBayesClassifier(train_model(train_set))
    
if __name__ == '__main__':
    main()