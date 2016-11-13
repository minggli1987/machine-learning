import nltk.classify.util
from nltk.classify import NaiveBayesClassifier
from nltk.corpus import movie_reviews as mr
from nltk.tokenize import word_tokenize
import string


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


print('train on {:.2f} instances, test on {:.2f} instances'.format(len(trainfeats), len(testfeats)))

classifier = NaiveBayesClassifier.train(trainfeats)
print('accuracy:', nltk.classify.util.accuracy(classifier, testfeats))

print(trainfeats)

classifier.show_most_informative_features()

bad_review = 'As ever, Paolo Sorrentino ironically cuts the legs out from under his protagonists\' wistfulness with grotesquerie'
good_review = '''Paolo Sorrentino, with Youth, delivers his most tender film to date, an emotionally rich contemplation of life's wisdom gained, lost and remembered - with cynicism harping from the sidelines, but as a wearied chord rather than a major motif.'''
madeup_review = '''
There are three levels of heart block, and usually only the most serious type causes symptoms. This type of heart block is known as a complete, or third-degree, heart block.
Symptoms of heart block can include: shortness of breath, palpitations, fainting
'''


def classify(review):

    assert isinstance(review, str), 'review must be string'

    # tokenize the review in lower case without punctuation

    cleansed_review = review.lower().translate(str.maketrans('', '', string.punctuation))
    tokenized_review = word_tokenize(cleansed_review, language='english')
    feats = dict([(word, True) for word in tokenized_review])
    print('\n', review, '\n', 'NLP classifer believes above review is:')
    print(feats)
    return classifier.classify(feats)


print(classify(madeup_review))
