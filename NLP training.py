import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews as mr


def word_feats(words):
    return dict([(word, True) for word in words])

negids = mr.fileids('neg')
posids = mr.fileids('pos')

negfeats = [(word_feats(mr.words(fileids=[f])), 'neg') for f in negids]
posfeats = [(word_feats(mr.words(fileids=[f])), 'pos') for f in posids]

negcutoff = int(len(negfeats) * 3 / 4)
poscutoff = int(len(posfeats) * 3 / 4)

trainfeats = negfeats[:negcutoff] + posfeats[:poscutoff]
testfeats = negfeats[negcutoff:] + posfeats[poscutoff:]
print ('train on {:.2f} instances, test on {:.2f} instances'.format(len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print ('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))

classifier.show_most_informative_features()
