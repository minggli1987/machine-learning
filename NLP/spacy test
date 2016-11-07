import numpy as np
import tensorflow as tf
import pandas as pd

from spacy.en import English


parser = English()

multiSentence = "There is an art, it says, or rather, a knack to flying." \
                 "The knack lies in learning how to throw yourself at the ground and miss." \
                 "In the beginning the Universe was created. This has made a lot of people "\
                 "very angry and been widely regarded as a bad move."

parsedData = parser(multiSentence)

# Let's look at the tokens
# All you have to do is iterate through the parsedData
# Each token is an object with lots of different properties
# A property with an underscore at the end returns the string representation
# while a property without the underscore returns an index (int) into spaCy's vocabulary
# The probability estimate is based on counts from a 3 billion word
# corpus, smoothed using the Simple Good-Turing method.


def summary_stats(par_data):

	for i, token in enumerate(parsedData):
		print("original:", token.orth, token.orth_)
		print("lowercased:", token.lower, token.lower_)
		print("lemma:", token.lemma, token.lemma_)
		print("shape:", token.shape, token.shape_)
		print("prefix:", token.prefix, token.prefix_)
		print("suffix:", token.suffix, token.suffix_)
		print("log probability:", token.prob)
		print("Brown cluster id:", token.cluster)
		print("----------------------------------------")
		if i > 1:
		    break
	pass

# Let's look at the sentences
sents = []
# the "sents" property returns spans
# spans have indices into the original string
# where each index value represents a token

for span in parsedData.sents:
    # go from the start to the end of each span, returning each token in the sentence
    # combine each token using join()
    sent = ''.join(parsedData[i].string for i in range(span.start, span.end)).strip()
    sents.append(sent)

for sentence in sents:
    print(sentence)

# Let's look at the part of speech tags of the first sentence
for span in parsedData.sents:
    sent = [parsedData[i] for i in range(span.start, span.end)]
    break

for token in sent:
    print(token.orth_, token.pos_)

# Let's look at the dependencies of this example:
example = "The boy with the spotted dog quickly ran after the firetruck."
parsedEx = parser(example)
# shown as: original token, dependency tag, head word, left dependents, right dependents
for token in parsedEx:
    print(token.orth_, token.dep_, token.head.orth_, [t.orth_ for t in token.lefts], [t.orth_ for t in token.rights])