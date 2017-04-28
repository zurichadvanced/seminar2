from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
import numpy as np
from sklearn.datasets import fetch_20newsgroups
from sklearn.svm import LinearSVC
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn import metrics


print("Loading 20 newsgroups dataset for categories:")

remove = ('headers', 'footers', 'quotes')
categories = ['rec.motorcycles', 'soc.religion.christian','comp.graphics', 'sci.med']
data_train = fetch_20newsgroups(subset='train', categories=categories,shuffle=True, random_state=42)

data_test = fetch_20newsgroups(subset='test', categories=categories,shuffle=True, random_state=42)

print('data loaded...')
print('Numbers of categories')
print(data_train.target_names)
print(len(data_train.data))
print("We have %d documents to train" % len(data_train.data))
print("We have %d documents to test" % len(data_test.data))

print('lets show the first documents of the data train')
print("------------------------------------------------------")
print("------------------------------------------------------")
print("\n".join(data_train.data[0].split("\n")[:60]))
print("------------------------------------------------------")
print("This first email has the label:  %s " % data_train.target_names[data_train.target[0]])

#--------------------------------------
#----Data Processing
#Text preprocessing, tokenizing and filtering of stopwords are included in a high level component that is able to build a dictionary of features and transform documents to feature vectors

count_vect = CountVectorizer()
X_train_counts = count_vect.fit_transform(data_train.data)
print(X_train_counts.shape)

#CountVectorizer supports counts of N-grams of words or consecutive characters. Once fitted, the vectorizer has built a dictionary of feature indices:
count_vect.vocabulary_.get(u'algorithm')


tf_transformer = TfidfTransformer(use_idf=False).fit(X_train_counts)
X_train_tf = tf_transformer.transform(X_train_counts)

tfidf_transformer = TfidfTransformer()
X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)

#---Training a classifier
#clf = MultinomialNB().fit(X_train_tfidf, data_train.target)
#clf = LinearSVC().fit(X_train_tfidf, data_train.target)

#---- Predict unknow documents
clf = LinearSVC().fit(X_train_tfidf, data_train.target)

docs_new = ['We are learning how to apply machine learning algorithm in text classification', 'Yamaha and Honda are the favourite for the next MotoGP championship','I have to take the medicine for my allergies in Spring']
X_new_counts = count_vect.transform(docs_new)
X_new_tfidf = tfidf_transformer.transform(X_new_counts)

predicted = clf.predict(X_new_tfidf)

for doc, category in zip(docs_new, predicted):
    print('%r => %s' % (doc, data_train.target_names[category]))


y_test = data_test.target
#Evaluation of the performance on the test set

print("\n---PERFORMANCE----")

#Naive Bayes
from sklearn.pipeline import Pipeline
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', MultinomialNB()),
])

text_clf = text_clf.fit(data_train.data, data_train.target)

docs_test = data_test.data
predicted = text_clf.predict(docs_test)
np.mean(predicted == data_test.target)

score = metrics.accuracy_score(y_test, predicted)
print("Accuracy Naive Bayes :   %0.3f" % score)


#--- Lets see if we can do better in SVM
from sklearn.linear_model import SGDClassifier
text_clf = Pipeline([('vect', CountVectorizer()),
                     ('tfidf', TfidfTransformer()),
                     ('clf', SGDClassifier(loss='hinge', penalty='l2',
                                           alpha=1e-3, n_iter=5, random_state=42)),
])
_ = text_clf.fit(data_train.data, data_train.target)
predicted = text_clf.predict(docs_test)
np.mean(predicted == data_test.target)
score = metrics.accuracy_score(data_test.target, predicted)

print("Accuracy SVM :   %0.3f" % score)
#---
