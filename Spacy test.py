import spacy

nlp = spacy.load('en')


texts = [u'One document.', u'...', u'Lots of documents']
# .pipe streams input, and produces streaming output
iter_texts = (texts[i % 3] for i in range(100000000))
for i, doc in enumerate(nlp.pipe(iter_texts, batch_size=50, n_threads=4)):
    assert doc.is_parsed
    if i == 100:
        break