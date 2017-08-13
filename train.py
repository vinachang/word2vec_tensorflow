import random
import numpy as np
from matplotlib import pylab
from sklearn.manifold import TSNE
from models.skip_gram import SkipGram
from models.cbow import CBOW

from data_prep import maybe_download, read_data

def show_closest_words(valid_examples, embeddings, reverse_dictionary, top_k=8):
    for i in range(valid_size):
        valid_example = valid_examples[i]
        valid_word = reverse_dictionary[valid_example]
          # number of nearest neighbors
        sim = embeddings.dot(embeddings[valid_example, :].T)
        nearest = (-sim).argsort()[1:top_k + 1]
        log = 'Nearest to %s:' % valid_word
        for k in range(top_k):
            close_word = reverse_dictionary[nearest[k]]
            log = '%s %s,' % (log, close_word)
        print(log)


def plot_embedding(embeddings, reverse_dictionary, num_points=400):
    labels = [reverse_dictionary[i] for i in range(1, num_points+1)]
    assert embeddings.shape[0] >= len(labels), 'More labels than embeddings'

    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    two_d_embeddings = tsne.fit_transform(embeddings[1:num_points+1, :])
    pylab.figure(figsize=(15, 15))  # in inches
    for i, label in enumerate(labels):
        x, y = two_d_embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(label, xy=(x, y), xytext=(5, 2), textcoords='offset points',
                       ha='right', va='bottom')
    pylab.show()

def show_result(model, valid_examples, top_k=8):
    show_closest_words(valid_examples, model.embeddings, model.reverse_dictionary, top_k=8)
    plot_embedding(model.embeddings, model.reverse_dictionary)

# We pick a random validation set to sample nearest neighbors. here we limit the
# validation samples to the words that have a low numeric ID, which by
# construction are also the most frequent.
  # Random set of words to evaluate similarity on.
  # Only pick dev samples in the head of the distribution.
valid_size = 16
valid_window = 100
valid_examples = np.array(random.sample(range(valid_window), valid_size))

# get training data
filename = maybe_download('text8.zip', 31344016)
words = read_data(filename)

# skip gram
skip_gram_model = SkipGram()
skip_gram_model.train(words)
show_result(skip_gram_model, valid_examples)


# skip gram
cbow_model = CBOW()
cbow_model.train(words)
show_result(cbow_model, valid_examples)
