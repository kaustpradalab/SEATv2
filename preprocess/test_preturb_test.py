#please run the code "prepare_data_w2v/bert" first.if you want to get the result in the last form in our paper, modify the comment section of "prepare_data_w2v/bert"
#this code is used to test the vectors.
from gensim.models import KeyedVectors
model_path='./.vector_cache/wiki.simple.vec'
word_vectors = KeyedVectors.load_word2vec_format(model_path, binary=False)
if word_vectors['word'] is not None:
    print("gensim load successed.")
else:
    print("gensim load failed.")
vector = word_vectors['word']
similar_words = word_vectors.most_similar('love', topn=1)
print("Vector for 'word':", vector)
print("Most similar words to 'love':", similar_words)