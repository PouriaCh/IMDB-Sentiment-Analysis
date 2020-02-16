import math
import random
import os
import csv
import numpy as np
from methods import *
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.preprocessing import Normalizer
from sklearn.naive_bayes import BernoulliNB
from sklearn import neighbors, tree
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import GridSearchCV
from nltk import word_tokenize

# Data preparation and partitioning

pos_root = 'train/pos/'
neg_root = 'train/neg/'
test_root = 'test/'

All_pos = os.listdir(pos_root)
All_neg = os.listdir(neg_root)
All_test = sorted(os.listdir(test_root), key=lambda x: int(os.path.splitext(x)[0]))

pos_corpus = Corpus(All_pos, pos_root)
neg_corpus = Corpus(All_neg, neg_root)

Np = len(pos_corpus)
Nt = 2 * Np
portion = 0.85
N_train = int(portion * Nt)
N_valid = int((1 - portion) * Nt)
train_corpus = [None] * Nt
Y_train = [None] * Nt

for i in range(len(pos_corpus) * 2):
    if i % 2 == 0:
        train_corpus[i] = pos_corpus[int(i / 2)]
        Y_train[i] = 1
    else:
        train_corpus[i] = neg_corpus[int(math.floor(i / 2))]
        Y_train[i] = 0

test_corpus = Corpus(All_test, test_root)

# Removing stop-words from Training set (comment if you don't want to delete stopwords)

train_corpus = delStopWords(train_corpus)

# Before splitting training set and validation set, we perform random shuffle

mixed_train = list(zip(train_corpus, Y_train))

random.shuffle(mixed_train)

train_corpus, Y_train = zip(*mixed_train)

X_training = train_corpus[:N_train]
Y_training = Y_train[:N_train]
X_validation = train_corpus[N_train:]
Y_validation = Y_train[N_train:]

# counting number of features

cw = CountVectorizer(ngram_range=(1,2),tokenizer=word_tokenize)
X_1 = cw.fit_transform(X_training)
print("Number of features: "+ str(X_1.shape[1]))


########################################################################################




########################################################################################
# Naive-Bayes in Scikit-Learn

# 1) using binary features

NB_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),tokenizer=LemmaTokenizer(),binary=True)),
                    ('clf', BernoulliNB())])

NB_clf = NB_clf.fit(X_training, Y_training)
Y_pred_NB = NB_clf.predict(X_validation)
Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation,Y_pred_NB)))/N_valid
print('Validation Accuracy for BernoulliNB with binary features is: ' + str(Accuracy))


# 2) using TfIDF

NB_clf = Pipeline([ ('tfidf', TfidfVectorizer(ngram_range=(1,2),tokenizer=LemmaTokenizer(),sublinear_tf=True)),
                    ('norm', Normalizer()),
                    ('clf', BernoulliNB())])

NB_clf = NB_clf.fit(X_training, Y_training)
Y_pred_NB = NB_clf.predict(X_validation)
Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation,Y_pred_NB)))/N_valid
print('Validation Accuracy for BernoulliNB with TfIDF is: ' + str(Accuracy))
########################################################################################

#1) using binary features

K = 34
KNN_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), tokenizer=LemmaTokenizer(), binary=True)),
                    ('clf', neighbors.KNeighborsClassifier(n_neighbors=K, weights='distance'))])
print(str(K) + "-NN pipeline created!")

KNN_clf.fit(X_training, Y_training)

print((str(K) + "-NN Model fitted!"))

Y_pred_KNN = KNN_clf.predict(X_validation)
KNN_Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation,Y_pred_KNN)))/N_valid
print('Validation Accuracy for ' + str(K) + '-NN with binary features is: '+ str(KNN_Accuracy))

# 2) using TfIDF

KNN_clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2), tokenizer=LemmaTokenizer(), sublinear_tf=True)),
                    ('norm', Normalizer()),
                    ('clf', neighbors.KNeighborsClassifier(n_neighbors= K, weights= 'distance'))])

print(str(K) + "-NN pipeline created!")

KNN_clf.fit(X_training, Y_training)

print((str(K) + "-NN Model fitted!"))

Y_pred_KNN = KNN_clf.predict(X_validation)
KNN_Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation,Y_pred_KNN)))/N_valid
print('Validation Accuracy for ' + str(K) + '-NN with TfIDF is: '+ str(KNN_Accuracy))

# KNN Pipeline
grs = input("Do you want to do a grid search for the best parameters of the KNN? (y/n)")

if grs:
    # KNN pipeline with Grid search

    KNN_parameters = {'clf__n_neighbors': list(range(23, 35))}

    GS_KNN_clf = GridSearchCV(KNN_clf, KNN_parameters, cv=3, iid=False, n_jobs=-1)
    GS_KNN_clf = GS_KNN_clf.fit(train_corpus, Y_train)

    print(GS_KNN_clf.best_params_)
    print(GS_KNN_clf.best_score_)
########################################################################################

# Decision Tree pipeline

# 1) using binary features

DT_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1,2),tokenizer=LemmaTokenizer(), binary=True)),
                    ('clf', tree.DecisionTreeClassifier())])

print("Decision Tree pipeline created!")

DT_clf.fit(X_training, Y_training)

print("Decision Tree Model fitted!")

Y_pred_DT = DT_clf.predict(X_validation)

DT_Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation,Y_pred_DT)))/N_valid
print('Validation Accuracy for Decision Tree with binary features is: '+ str(DT_Accuracy))


# 2) using TfIDF

DT_clf = Pipeline([('tfidf', TfidfVectorizer(ngram_range=(1,2),tokenizer=LemmaTokenizer(),sublinear_tf=True)),
                    ('norm', Normalizer()),
                    ('clf', tree.DecisionTreeClassifier())])

print("Decision Tree pipeline created!")

DT_clf.fit(X_training, Y_training)

print("Decision Tree Model fitted!")

Y_pred_DT = DT_clf.predict(X_validation)

DT_Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation,Y_pred_DT)))/N_valid
print('Validation Accuracy for Decision Tree with TfIDF is: '+ str(DT_Accuracy))


########################################################################################
# SVM pipeline

# 1) using binary features

SVM_clf = Pipeline([('vect', CountVectorizer(ngram_range=(1, 2), tokenizer=LemmaTokenizer(), binary=True)),
                    ('clf', SGDClassifier(loss='squared_hinge', alpha=1e-4, max_iter=70, tol=0.18))])

print("SVM pipeline created!")

SVM_clf.fit(X_training, Y_training)

print("SVM Model fitted!")

Y_pred_SVM = SVM_clf.predict(X_validation)

SVM_Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation, Y_pred_SVM))) / N_valid
print('Validation Accuracy for SVM with binary features is: ' + str(SVM_Accuracy))

# 2) using TfIDF

SVM_clf = Pipeline([('tfidf', TfidfVectorizer(sublinear_tf=True, ngram_range=(1, 2), tokenizer=LemmaTokenizer())),
                    ('norm', Normalizer()),
                    ('clf', SGDClassifier(loss='squared_hinge', alpha=1e-4, max_iter=70, tol=0.18))])

print("SVM pipeline created!")

SVM_clf.fit(X_training, Y_training)

print("SVM Model fitted!")

Y_pred_SVM = SVM_clf.predict(X_validation)

SVM_Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation, Y_pred_SVM))) / N_valid
print('Validation Accuracy for SVM with TfIDF is: ' + str(SVM_Accuracy))

# SVM Grid Search
SVM_parameters = {'vect__tokenizer': [LemmaTokenizer(),word_tokenize]}

GS_SVM_clf = GridSearchCV(SVM_clf, SVM_parameters, cv=5, iid=False, n_jobs=-1) # cv = k in k-fold
GS_SVM_clf = GS_SVM_clf.fit(train_corpus, Y_train)
print(GS_SVM_clf.best_params_)
# Y_pred_GS_SVM = SVM_clf.predict(X_validation)
# GS_SVM_Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation,Y_pred_GS_SVM)))/N_valid
print('Best configuration score for SVM is: '+ str(GS_SVM_clf.best_score_))

grs = input("Do you want to do a grid search for the best parameters of the KNN? (y/n)")
if grs:
    # SVM Grid Search
    SVM_parameters = {'vect__tokenizer': [LemmaTokenizer(), word_tokenize]}

    GS_SVM_clf = GridSearchCV(SVM_clf, SVM_parameters, cv=5, iid=False, n_jobs=-1)  # cv = k in k-fold
    GS_SVM_clf = GS_SVM_clf.fit(train_corpus, Y_train)
    print(GS_SVM_clf.best_params_)
    # Y_pred_GS_SVM = SVM_clf.predict(X_validation)
    # GS_SVM_Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation,Y_pred_GS_SVM)))/N_valid
    print('Best configuration score for SVM is: ' + str(GS_SVM_clf.best_score_))

########################################################################################

# Logistic Regression pipeline

# 1) using binary features

LogReg_clf = Pipeline([('vect', CountVectorizer(tokenizer=LemmaTokenizer(), ngram_range=(1,2), binary=True)),
                       ('clf', LogisticRegression(solver='lbfgs', multi_class='ovr', max_iter=500))])

print("Logestic Regression pipeline created!")
LogReg_clf.fit(X_training, Y_training)

print("Model fitted!")

Y_pred_LogReg = LogReg_clf.predict(X_validation)

LogReg_Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation,Y_pred_LogReg)))/N_valid
print('Validation Accuracy for Logistic Regression with binary features is: '+ str(LogReg_Accuracy))


# 2) using TfIDF

LogReg_clf = Pipeline([('tfidf', TfidfVectorizer(tokenizer= LemmaTokenizer(), ngram_range=(1,2), sublinear_tf=True)),
                       ('norm', Normalizer()),
                       ('clf', LogisticRegression(random_state=0, solver='lbfgs', multi_class='ovr', max_iter=200))])

print("Logestic Regression pipeline created!")
LogReg_clf.fit(X_training, Y_training)

print("Model fitted!")

Y_pred_LogReg = LogReg_clf.predict(X_validation)

LogReg_Accuracy = np.sum(np.logical_not(np.logical_xor(Y_validation,Y_pred_LogReg)))/N_valid
print('Validation Accuracy for Logistic Regression with TfIDF is: '+ str(LogReg_Accuracy))

########################################################################################
# In case you want to export the predictions, choose your desired classifier and run the following lines
# Testing on Test set and creating CSV file for Kaggle

#best_predictor = SVM_clf.predict(test_corpus)
#csvWriter(best_predictor,8)

#### This is the end of the code