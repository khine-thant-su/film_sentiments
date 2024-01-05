import pandas as pd
import string
import re
import nltk
import random
import pickle
from nltk.corpus import stopwords
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from zipfile import ZipFile

PATH_TO_DATA_FILE = 'movie.csv'
TEXT_COLUMN_NAME = 'text'
LABEL_COLUMN_NAME = 'label'
DATA_LENGTH = 2000
NEGATIVE_SENTIMENT_LABEL = 0
POSITIVE_SENTIMENT_LABEL = 1
NEGATIVE_SENTIMENT = 'neg'
POSITIVE_SENTIMENT = 'pos'
STOPWORD_TYPE = 'english'
FILE_NAME = 'lr_classifier'


class LogisticRegressionClassifier:
    def __init__(self, classifier): 
        self.classifier = classifier 
    
    @classmethod
    def from_pickled(cls, path):
        '''Unpickles the file given by path, returns an instance of the class initialized with the classifier object.'''
        
        classifier = pickle.load(open(path,'rb'))
        return cls(classifier)
    
    def test(self, test_features, test_sentiments):
        '''Evaluates the model on test data, and returns the accuracy.'''
        
        return classifier.score(X_test, y_test)
    
    def predict_class(self, teststring):
        '''Takes in a string, uses the TfidfVectorizer to transform the string into a numerical vector, 
        and returns a prediction for the sentiment of that string using a Logistic Regression classifier.
            
         Args:
         teststring(str)'''
        
        test_doclist = [clean(teststring)] 
        documents = compile_docs(main.df, TEXT_COLUMN_NAME)
        
        vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))   
        
        train_matrix = vectorizer.fit_transform(documents) # Fitting the TF-IDF vectorizer to the data
        
        test_matrix = vectorizer.transform(test_doclist) # Vectorizing the input documents
        test_array = test_matrix.toarray()
        
        classifier = train_model(main.train_text, main.train_label)

        return classifier.predict(test_array)  
        
    
    def store(self, path):
        '''Accepts a file path and stores the model at that file path using the pickle library.'''
        
        with open(path,'wb') as f:
            pickle.dump(classifier, f)

            
def clean(text): 
    '''Converts the input text into lower case, removes <br> tags, punctuation, whitespace, and stopwords. Lemmatizes the words.
    Returns the processed words in a list.
    
        Args:
        text(str): input text'''
    
    wnl = WordNetLemmatizer()
    text = "".join([i.lower() for i in text.replace('<br', '') if i not in string.punctuation])
    text = re.sub('\s+',' ', text)
    text = [word for word in text.split() if word not in stopwords.words('english')]
    text = ' '.join([wnl.lemmatize(word) for word in text])
    
    return text

def compile_docs(data, col_name):
    '''Cleans strings from the specified column from data, appends the cleaned strings to a list, returns the list of cleaned strings.
        
        Args:
        data(df): a dataframe
        col_name(str): name of text column from the dataframe
    '''
    
    documents = []
    for i in range(len(data)):
        doc = data.loc[i, col_name]
        processed_doc = clean(doc)
        documents.append(processed_doc)
        
    return documents

def tfidf_vectorize(docs):
    '''Transforms text data into a matrix of TF-IDF features, using word as analyzer, and extracting both unigrams and bigrams.
    Returns a document-term matrix.
    
        Args:
        docs: a list (or another iterable) that generates strings'''
    
    vectorizer = TfidfVectorizer(analyzer='word', ngram_range=(1, 2))
    tfidf_matrix = vectorizer.fit_transform(docs) # tfidf_matrix = csr.csr_matrix
    
    return tfidf_matrix

def train_model(tfidf_arrays, labels):
    '''Trains the model, returns the model.
    
        Args:
        tfidf_arrays(array): arrays created by transforming text into a TF-IDF document-term matrix
        labels(array): sentiment labels in the form of an array'''
    
    classifier = LogisticRegression().fit(tfidf_arrays, labels) 
    
    return classifier


def main():
    with ZipFile('imdb ratings.zip') as zf:
        f = zf.open(PATH_TO_DATA_FILE)
    
    df = pd.read_csv(f)
    smalldf = df[:DATA_LENGTH]
    smalldf = smalldf.drop_duplicates(ignore_index = True)
    smalldf.loc[smalldf[LABEL_COLUMN_NAME] == NEGATIVE_SENTIMENT_LABEL, LABEL_COLUMN_NAME] = NEGATIVE_SENTIMENT
    smalldf.loc[smalldf[LABEL_COLUMN_NAME] == POSITIVE_SENTIMENT_LABEL, LABEL_COLUMN_NAME] = POSITIVE_SENTIMENT
    
    main.df = smalldf
    
    documents = compile_docs(smalldf, TEXT_COLUMN_NAME)
        
    X = tfidf_vectorize(documents).toarray()
    y = smalldf[LABEL_COLUMN_NAME].values
    
    main.train_text, main.test_text, main.train_label, main.test_label = train_test_split(X, y)
    
    classifier = train_model(main.train_text, main.train_label)
    
    model_accuracy = classifier.score(main.test_text, main.test_label)
    
    print('LR model accuracy:', model_accuracy, '\n')
    
    with open(FILE_NAME,'wb') as f:
        pickle.dump(classifier, f)
  
    model = LogisticRegressionClassifier(classifier)
    
    teststring = "It is insulting that you think I am not smart enough."
    print("Input: ", teststring, '\n')
    print("Prediction of input: ", model.predict_class(teststring))
    
if __name__ == '__main__':
    main()
      