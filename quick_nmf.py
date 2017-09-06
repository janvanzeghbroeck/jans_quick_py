import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.stop_words import ENGLISH_STOP_WORDS
from sklearn.decomposition import NMF
from collections import Counter
from sklearn.decomposition import PCA
# from sklearn.decomposition import TruncatedSVD

class QuickNMF(object):
    def __init__(self,k = 5):
        '''
        A quick and simple way to use NMF on text data written by Jan Van Zeghbroeck

        https://github.com/janvanzeghbroeck/jans_quick_py

        k = 5
            --> type int
            --> number of clusters
        '''
        self.k = k
        np.random.seed(42)

    def Fit_transform(self, text, labels, stop_words =[], vectorizer = None, **kwargs):
        '''
        Fits and transforms the data
        '''
        self.Fit(text, labels, stop_words =stop_words, vectorizer = vectorizer)
        self.Transform(**kwargs)

    def Fit(self, text, labels, stop_words =[], vectorizer = None):
        '''
        Fits the data into the class and does a TFIDF on the raw text

        text = None
            --> type iterable (list or np.array is best) of strings
            --> len(text) must equal len(labels)
        labels
            --> type iterable of strings
            --> labels for each text
            --> len(labels) must equal len(text)
        stop_words = None
            --> type is list of strings
            --> words that you dont want as part of the nmf model
        vectorizer
            --> a seperate sklearn tfidf model to use instead of the default

        Creates:
            self.raw_text == the input text known as a list of documents
            self.labels == the input labels
            self.vectorizer == tfidf sklearn model
            self.processed_text == processed text
            self.bag == bag of words / features of self.processed_text / all the words that the model recognizes
        '''
        self.raw_text = np.array(text)
        self.labels = np.array(labels)

        if vectorizer == None: # default vectorizer (TFIDF)
            stopwords = set(list(ENGLISH_STOP_WORDS) + stop_words)
            vectorizer = TfidfVectorizer(ngram_range=(1, 2), max_df=.8, min_df=.2, stop_words = stopwords, max_features = 10000)
        self.vectorizer = vectorizer
        X = vectorizer.fit_transform(text) # this is the fit_transform from sklearn's tfidf not from this class
        self.processed_text = X
        self.bag = np.array(vectorizer.get_feature_names())

    def Transform(self,**kwargs):
        '''
        Transforms the fit tfidf data into nmf topics

        **kwargs will cary over to the sklearn NMF model

        Creates
            self.w = labels connection with each cluster. shape = (len(labels),k)
            self.h = words in bag (features) connection with each cluster. shape = (k, len(self.bag))
            self.model = the sklearn nmf model
            self.topic_labels = topic label for each text document
        '''
        model = NMF(n_components = self.k, **kwargs)

        model.fit(self.processed_text) # fit from sklearn nmf class
        self.w = model.transform(self.processed_text)
        self.h = model.components_
        self.model = model
        self.topic_labels = np.argmax(self.w,axis = 1)

    def Print_tops(self,top=10):
        '''
        Prints the top words and labels for each topic. usefull for visualizing what topics mean

        top = 10
            --> type = int
            --> prints the top this many items for each cluster

        returns
            --> top_words = the top words associated with each topic
        '''
        top_words = []
        for group in range(self.k):
            #idx of the top ten words for each group
            i_words = np.argsort(self.h[group,:])[::-1][:top]
            words = self.bag[i_words]
            top_words.append(words) #top words for each cluster

            i_label = np.argsort(self.w[:,group])[::-1][:top]

            print ('-'*10)
            print ('Group:',group)
            print ('WORDS')
            for word in words:
                print ('-->',word)
            print ('LABELS')
            for i in i_label:
                print ('==>',labels[i])

        return top_words

    def Plot_pca(self, show_plot = True):
        '''
        Makes a pretty pca plot using matplotlib

        show_plot = True
            --> type = boolean
            --> True shows the plot with plt.show, False doesnt

        Creates:
            self.pca = sklearn pca model
            self.x_pca = pca feature matrix of self.w
        '''
        colors = ['steelblue', 'purple', 'green', 'midnightblue', '.2', 'c', 'm', 'yellow', 'red', '.8', 'blue', 'orange', '.5', 'pink', 'black'] # can add more colors
        colors = colors + colors # doubles the list

        plt.style.use('fivethirtyeight') # to style the plot

        # make the PCA in 3 dim
        pca = PCA(n_components=3)
        pca.fit(self.w)
        X_pca = pca.transform(self.w)
        self.pca = pca
        self.x_pca = X_pca
        print ('\nTotal Explained Variance in 3 Dimensions =', np.sum(pca.explained_variance_ratio_))

        plt.figure(figsize = (10,6))

        # plot each cluster a different color and add size array
        for i in range(self.k):
            idx = np.argwhere(self.topic_labels==i)
            X_ = X_pca[idx][:,0]

            plt.scatter(X_[:,0],X_[:,1], s = X_[:,2]*1000+267, c = colors[i], alpha = .5, label = 'cluster {}'.format(i))

        # formatting the plot
        plt.title('PCA Scatter Plot (explained variance = {})'.format( round(np.sum(pca.explained_variance_ratio_),3) ))
        plt.ylabel('2nd PCA dimension (dot size is 3rd PCA dimension)', fontsize = 12)
        plt.xlabel('1st PCA dimension (dot size is 3rd PCA dimension)', fontsize = 12)
        plt.legend()
        plt.tight_layout()

        # show the plot
        if show_plot:
            plt.show()

if __name__ == '__main__':
    plt.close('all') # close any previous plot

    # Get the data prepared
    df = pd.read_pickle('data/new_beer_features.pkl')

    text = df['all_text'].apply(lambda x: ' '.join(x)).values
    labels = np.array(df.index.tolist())

    drop_lst = ['aroma', 'note', 'notes', 'sl', 'slight', 'light', 'hint', 'bit', 'little', 'lot', 'touch', 'character', 'some', 'something', 'retro', 'thing', ' ']

    # start of the class
    nmf = QuickNMF(k = 5)
    nmf.Fit_transform(text, labels, stop_words = drop_lst)
    top_words = nmf.Print_tops()
    nmf.Plot_pca(show_plot = True)
