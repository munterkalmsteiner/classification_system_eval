from gensim.corpora.wikicorpus import WikiCorpus
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import word2vec
from pprint import pprint
import multiprocessing
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

assert word2vec.FAST_VERSION > -1

wiki = WikiCorpus("/mnt/8tb_hdd/users/mun/corpora/wikipedia/enwiki-latest-pages-articles.xml.bz2")

class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument(content, [title])

documents = TaggedWikiDocument(wiki)
cores = multiprocessing.cpu_count()
model = Doc2Vec(dm=0, dbow_words=1, vector_size=200, window=8, min_count=10, epochs=10, workers=cores)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
model.save("wikipedia_20210308")
