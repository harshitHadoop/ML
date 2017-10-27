# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 12:30:58 2017

@author: harshit.karnata
"""

from sklearn.learning_curve import learning_curve
import pandas as pd
import calendar
import string
import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import RandomizedLogisticRegression
from sklearn.model_selection import cross_val_predict
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn import cross_validation
from sklearn import svm
from datetime import timedelta
from nltk.stem.snowball import SnowballStemmer
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
from sklearn import metrics
from sklearn.metrics import mean_squared_error,confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import log_loss
from imblearn.over_sampling import SMOTE 
from imblearn.over_sampling import ADASYN
from sklearn.linear_model import RandomizedLasso
from sklearn.preprocessing import MinMaxScaler,StandardScaler
from sklearn import preprocessing
from sklearn.feature_selection import RFE
from sklearn.svm import SVC
from sklearn.model_selection import StratifiedKFold
from sklearn.feature_selection import RFECV
from sklearn.datasets import load_digits
from sklearn.svm import SVR
from sklearn.svm import LinearSVC
from sklearn.preprocessing import normalize
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.feature_extraction.text import CountVectorizer
from scipy.sparse import hstack
import xgboost as xgb
from xgboost.sklearn import XGBClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.grid_search import GridSearchCV
import re
from sklearn.ensemble import RandomForestRegressor
import math

def stem_tokens(tokens, stemmer):
    stemmed = []
    for item in tokens:
        stemmed.append(stemmer.stem(item))
    return stemmed
def text_process(mess):
    '''
    remove punc
    remove stop words
    return a list of clean words
    
    '''
    mess1 = re.sub("[^a-zA-Z]"," ",mess)
    nopunc = [char for char in mess1 if char  not in string.punctuation]

    #stemming part
    stemmer=SnowballStemmer("english")
    nopunc = ''.join(nopunc)    
    
    stops=(stopwords.words('english'))
    a= [word for word in nopunc.split() if word.lower() not in stops]
    stems = stem_tokens(a,stemmer)
    return stems

data=pd.read_csv(r"C:\Users\harshit.karnata.NOTEBOOK436.000\Desktop\ocr1.csv",sep=',')

data1 = data[data.page_type == 'cheque']
temp = data1.groupby('check_checkNumber')['row_string'].apply(lambda col: ' '.join(col))
temp = pd.Series.to_frame(temp)
print("hello")
