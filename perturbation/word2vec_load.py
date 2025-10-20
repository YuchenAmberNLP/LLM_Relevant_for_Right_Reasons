import os
import gensim.downloader as api
from gensim.models import KeyedVectors

os.environ['GENSIM_DATA_DIR'] = "./word2vec"


print("Downloading and loading Word2Vec (Google News)...")
w2v = api.load("word2vec-google-news-300")
print("Loaded successfully.")