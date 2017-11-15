import pretraitements as pt
import pandas as pd
from time import time

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_validate
from sklearn import metrics

data_dir = "../data/"

print('#' * 80)
print("CLASSIFICATION DES TWEETS PAR POLARITE")
print()

# loading data
print("Chargement des données...")
train_data = pd.read_csv(data_dir + "task1-train.csv", sep='\t',
                         skiprows=10, header=None, names=['id', 'content', 'label'])
test_data = pd.read_csv(data_dir + "task1-test.csv", sep='\t',
                        skiprows=10, header=None, names=['id', 'content'])

print()

content_train = list(train_data['content'])
content_test = list(test_data['content'])
categories = set(train_data['label'])

print("Entrainement sur %d tweets" % (len(content_train)))
print("Test sur %d tweets" % (len(content_test)))
print("%d catégories" % (len(categories)))
print()

# pretraitements
for i, tweet in enumerate(content_train):
    content_train[i] = pt.removeURL(content_train[i])
    content_train[i] = pt.removeRallongements(content_train[i])

y_train = train_data['label']
#y_test = test_data['label']

print("Extraction des descripteurs par vectorisation tf-idf")
t0 = time()
# jusqu'aux bigrammes
vectorizer = TfidfVectorizer(ngram_range=(
    1, 2), tokenizer=pt.lemmatise, stop_words=pt.stopwords, sublinear_tf=True)
X_train = vectorizer.fit_transform(content_train)

# jusqu'aux trigrammes
vectorizer_3 = TfidfVectorizer(ngram_range=(
    1, 3), tokenizer=pt.lemmatise, stop_words=pt.stopwords, sublinear_tf=True)
X_3_train = vectorizer_3.fit_transform(content_train)

duration = time() - t0
print("Terminé en %fs" % duration)
print("Nb de tweets : {}, nb de descripteurs : {} (bigrammes) {} (trigrammes)".format(
    X_train.shape[0], X_train.shape[1], X_3_train.shape[1]))
print()

print("Extraction des descripteurs pour le test")
t0 = time()
X_test = vectorizer.transform(content_test)
X_3_test = vectorizer_3.transform(content_test)
duration = time() - t0

print("Terminé en %fs" % duration)
print("Nb de tweets : {}, nb de descripteurs : {} (bigrammes) {} (trigrammes)".format(
    X_test.shape[0], X_test.shape[1], X_3_test.shape[1]))
print()

descripteurs = vectorizer.get_feature_names()

scores = []

print("Selection des meilleurs descripteurs avec la méthode du chi-carré")

t0 = time()

# selection pour NaiveBayes
ch2_NB = SelectKBest(chi2, k=14000)
X_NB_train = ch2_NB.fit_transform(X_train, y_train)
X_NB_test = ch2_NB.transform(X_test)

#descripteurs_NB = [descripteurs[i] for i in ch2_NB.get_support(indices=True)]

# Selection pour NaiveBayes avec des trigrammes
ch2_NB3 = SelectKBest(chi2, k=28000)
X_NB3_train = ch2_NB3.fit_transform(X_3_train, y_train)
X_NB3_test = ch2_NB3.transform(X_3_test)

#descripteurs_NB3 = [descripteurs[i] for i in ch2_NB3.get_support(indices=True)]

# Selection pour LinearSVC
ch2_SVC = SelectKBest(chi2, k=11000)
X_SVC_train = ch2_SVC.fit_transform(X_train, y_train)
X_SVC_test = ch2_SVC.transform(X_test)

#descripteurs_SVC = [descripteurs[i] for i in ch2_SVC.get_support(indices=True)]

duration = time() - t0
print("Terminé en %fs" % duration)
print()

# TRAIN
print('-' * 80)
print('Entrainement des classifieurs')
print()

scoring = ['precision_macro', 'recall_macro', 'precision_micro']

print('NaiveBayes avec bigrammes')
classifier_NB = MultinomialNB(alpha=.01)
classifier_NB.fit(X_NB_train, y_train)
#scores = cross_validate(classifier_NB, X_NB_train, y_train,
#                        scoring=scoring, return_train_score=False, cv=10)

#print(sorted(scores.keys()))
#print('precision_micro : ' + str(scores['test_precision_micro'].mean()))
#print('recall_macro : ' + str(scores['test_recall_macro'].mean()) +
#      ',   precision_macro : ' + str(scores['test_precision_macro'].mean()))
print()

print('NaiveBayes avec trigrammes')
classifier_NB3 = MultinomialNB(alpha=.01)
classifier_NB3.fit(X_NB3_train, y_train)
#scores = cross_validate(classifier_NB3, X_NB3_train, y_train,
#                        scoring=scoring, return_train_score=False, cv=10)

#print(sorted(scores.keys()))
#print('precision_micro : ' + str(scores['test_precision_micro'].mean()))
#print('recall_macro : ' + str(scores['test_recall_macro'].mean()) +
#      ',   precision_macro : ' + str(scores['test_precision_macro'].mean()))
print()

print('LinearSVC avec bigrammes')
classifier_SVC = LinearSVC()
classifier_SVC.fit(X_SVC_train, y_train)
#scores = cross_validate(classifier_SVC, X_SVC_train, y_train,
#                       scoring=scoring, return_train_score=False, cv=10)

#print(sorted(scores.keys()))
#print('precision_micro : ' + str(scores['test_precision_micro'].mean()))
#print('recall_macro : ' + str(scores['test_recall_macro'].mean()) +
#     ',   precision_macro : ' + str(scores['test_precision_macro'].mean()))
print()

#plt.plot(range(1, X_train.shape[1], 1000), scores)
# plt.grid(True)
# plt.show()

# PREDICTION
print('-' * 80)
print('Test des classifieurs sur le corpus de test')
print()

pred_NB = classifier_NB.predict(X_NB_test)
pred_NB3 = classifier_NB3.predict(X_NB3_test)
pred_SVC = classifier_SVC.predict(X_SVC_test)

run1 = pd.DataFrame(pred_NB, index=test_data['id'])
run2 = pd.DataFrame(pred_NB3, index=test_data['id'])
run3 = pd.DataFrame(pred_SVC, index=test_data['id'])

run1.to_csv("task1-run1-equip3.csv", sep='\t', header=False)
run2.to_csv("task1-run2-equip3.csv", sep='\t', header=False)
run3.to_csv("task1-run3-equip3.csv", sep='\t', header=False)


# meilleur chi pour NB : 14 000
# meilleur chi pour NB 1-3 : 28 000
# meilleur chi pour LinearSVC : 11 000
# meilleur chi DecisionTree : 9 000
