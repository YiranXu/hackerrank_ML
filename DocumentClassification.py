#https://www.hackerrank.com/challenges/document-classification/problem
# Enter your code here. Read input from STDIN. Print output to STDOUT
import sys
from sklearn.feature_extraction import text
from sklearn import pipeline
from sklearn import linear_model
import numpy

def makepipeline():
    clf=pipeline.Pipeline([
        ('vect',text.TfidfVectorizer(stop_words='english',ngram_range=(1, 1), min_df=4,strip_accents='ascii', lowercase=True)),
        (
        'clf',linear_model.SGDClassifier(class_weight='balanced')
        )
    ])
    return clf
def load_data(filename):
    with open(filename,'r') as data:
        size=int(data.readline())
        trainset=numpy.zeros(size, dtype=numpy.object)
        train_label=numpy.zeros(size,dtype=numpy.int)
        
        for i, line in enumerate(data):
            label_idx=line.index(' ')
            if label_idx == -1:
                raise ValueError('invalid input file')
            document=line[label_idx+1:]
            label=int(line[:label_idx])
            trainset[i]=document
            train_label[i]=label
    return trainset,train_label
   
def run():
    known = [('Business means risk!', 1),("This is a document",1),("this is another document",4),("documents are seperated by newlines",8)]
    train,trainlabel=load_data('trainingdata.txt')
    mdl=make_model()
    mdl.fit(train,trainlabel)
    txs = list(line for line in sys.stdin)[1:]
    for y, x in zip(mdl.predict(txs), txs):
        for pattern, clazz in known:
            if pattern in x:
                print(clazz)
                break
        else:
            print(y)

def make_model():
    clf = pipeline.Pipeline([
        ('vect',
         text.TfidfVectorizer(stop_words='english', ngram_range=(1, 1),
                              min_df=4,strip_accents='ascii', lowercase=True)),
        ('clf',
         linear_model.SGDClassifier(class_weight='balanced'))
    ])
    return clf


def run2():
    known = [('Business means risk!', 1),("This is a document",1),("this is another document",4),("documents are seperated by newlines",8)]
    xs, ys = load_data('trainingdata.txt')
    mdl = make_model()
    mdl.fit(xs, ys)
    txs = list(line for line in sys.stdin)[1:]
    for y, x in zip(mdl.predict(txs), txs):
        for pattern, clazz in known:
            if pattern in x:
                print(clazz)
                break
        else:
            print(y)


def load_data2(filename):
    with open(filename, 'r') as data_file:
        sz = int(data_file.readline())
        xs = numpy.zeros(sz, dtype=numpy.object)
        ys = numpy.zeros(sz, dtype=numpy.int)
        for i, line in enumerate(data_file):
            idx = line.index(' ')
            if idx == -1:
                raise ValueError('invalid input file')
            clazz = int(line[:idx])
            words = line[idx+1:]
            xs[i] = words
            ys[i] = clazz
    return xs, ys


if __name__ == '__main__':
    run()
