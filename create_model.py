import gensim.corpora.wikicorpus as wc
from gensim.corpora.dictionary import Dictionary
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from gensim.models import word2vec
from gensim import utils
from pprint import pprint
import multiprocessing
import logging
logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

assert word2vec.FAST_VERSION > -1

# Sources
# https://dumps.wikimedia.org/enwiki/
# https://dumps.wikimedia.org/svwiki/


def myinit(self, fname, processes=None, lemmatize=utils.has_pattern(), dictionary=None,
             filter_namespaces=('0',), tokenizer_func=wc.tokenize, article_min_tokens=wc.ARTICLE_MIN_WORDS,
             token_min_len=wc.TOKEN_MIN_LEN, token_max_len=wc.TOKEN_MAX_LEN, lower=True, filter_articles=None):
    """Initialize the corpus.

        Unless a dictionary is provided, this scans the corpus once,
        to determine its vocabulary.

        Parameters
        ----------
        fname : str
            Path to the Wikipedia dump file.
        processes : int, optional
            Number of processes to run, defaults to `max(1, number of cpu - 1)`.
        lemmatize : bool
            Use lemmatization instead of simple regexp tokenization.
            Defaults to `True` if you have the `pattern <https://github.com/clips/pattern>`_ package installed.
        dictionary : :class:`~gensim.corpora.dictionary.Dictionary`, optional
            Dictionary, if not provided,  this scans the corpus once, to determine its vocabulary
            **IMPORTANT: this needs a really long time**.
        filter_namespaces : tuple of str, optional
            Namespaces to consider.
        tokenizer_func : function, optional
            Function that will be used for tokenization. By default, use :func:`~gensim.corpora.wikicorpus.tokenize`.
            If you inject your own tokenizer, it must conform to this interface:
            `tokenizer_func(text: str, token_min_len: int, token_max_len: int, lower: bool) -> list of str`
        article_min_tokens : int, optional
            Minimum tokens in article. Article will be ignored if number of tokens is less.
        token_min_len : int, optional
            Minimal token length.
        token_max_len : int, optional
            Maximal token length.
        lower : bool, optional
             If True - convert all text to lower case.
        filter_articles: callable or None, optional
            If set, each XML article element will be passed to this callable before being processed. Only articles
            where the callable returns an XML element are processed, returning None allows filtering out
            some articles based on customised rules.

        Warnings
        --------
        Unless a dictionary is provided, this scans the corpus once, to determine its vocabulary.

    """
    self.fname = fname
    self.filter_namespaces = filter_namespaces
    self.filter_articles = filter_articles
    self.metadata = False
    if processes is None:
        processes = max(1, multiprocessing.cpu_count() - 1)
    self.processes = processes
    self.lemmatize = lemmatize
    self.tokenizer_func = tokenizer_func
    self.article_min_tokens = article_min_tokens
    self.token_min_len = token_min_len
    self.token_max_len = token_max_len
    self.lower = lower

    if dictionary is None:
        #<<< Monkey patch is here >>>
        #self.dictionary = Dictionary(self.get_texts())
        self.dictionary = Dictionary(self.get_texts(), prune_at=None)
    else:
        self.dictionary = dictionary


#Monkey-patching WikiCorpus __init__ method so that the vocabulary in the dictionary does not get
#pruned at the default 2M tokens.
wc.WikiCorpus.__init__ = myinit

wiki = wc.WikiCorpus("/mnt/8tb_hdd/users/mun/corpora/wikipedia/20210308_enwiki-latest-pages-articles.xml.bz2", token_max_len=30)

class TaggedWikiDocument(object):
    def __init__(self, wiki):
        self.wiki = wiki
        self.wiki.metadata = True
    def __iter__(self):
        for content, (page_id, title) in self.wiki.get_texts():
            yield TaggedDocument(content, [title])

documents = TaggedWikiDocument(wiki)
cores = multiprocessing.cpu_count()
model = Doc2Vec(dm=0, dbow_words=1, vector_size=200, window=8, min_count=10, epochs=30, workers=cores)
model.build_vocab(documents)
model.train(documents, total_examples=model.corpus_count, epochs=model.iter)
model.save("wikipedia_en_20210308")
