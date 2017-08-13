import tensorflow as tf
import numpy as np
import random
import collections
import math
from data_prep import build_dataset

class SkipGram:
    """
    Word2Vec model trained with skip-gram.

    Parameters
    ----------
        batch_size: int
            how many training data to use in each step

        embedding_size: int
            dimension of the embedding vector

        window_size: int
            How many words to consider left and right

        num_skips: int
            How many times to reuse an input to generate a label

        num_sampled: int
            Number of negative examples to sample

    Attributes
    ----------
        embeddings: array, shape = [vocabulary_size, embedding_size]

        dictionary: dict
            A mapping of word to embeddings row indices.

        reverse_dictionary: dict
            A mapping of embeddings row indices to word
    """
    def __init__(self, batch_size=128, embedding_size=128, window_size=1,
                 num_skips=2, num_sampled=64, vocabulary_size=50000):
        assert batch_size % num_skips == 0
        assert num_skips <= 2 * window_size
        self.batch_size = batch_size
        self.embedding_size = embedding_size
        self.window_size = window_size
        self.num_skips = num_skips
        self.num_sampled = num_sampled
        self.vocabulary_size = vocabulary_size

        self.graph = tf.Graph()
        self._define_graph()

    @property
    def embeddings(self):
        return self.embeddings_

    @property
    def dictionary(self):
        return self.dictionary_

    @property
    def reverse_dictionary(self):
        return self.reverse_dictionary_

    def _generate_batch(self, data):
        batch = np.ndarray(shape=(self.batch_size), dtype=np.int32)
        labels = np.ndarray(shape=(self.batch_size, 1), dtype=np.int32)
        span = 2 * self.window_size + 1  # [ window_size target window_size ]
        buffer = collections.deque(maxlen=span)
        for _ in range(span):
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        for i in range(self.batch_size // self.num_skips):
            target = self.window_size  # target label at the center of the buffer
            targets_to_avoid = [self.window_size]
            for j in range(self.num_skips):
                while target in targets_to_avoid:
                    target = random.randint(0, span - 1)
                targets_to_avoid.append(target)
                batch[i * self.num_skips + j] = buffer[self.window_size]
                labels[i * self.num_skips + j, 0] = buffer[target]
            buffer.append(data[self.data_index])
            self.data_index = (self.data_index + 1) % len(data)
        return batch, labels

    def _define_graph(self):
        with self.graph.as_default(), tf.device('/cpu:0'):
            # Input data.
            self.train_dataset = tf.placeholder(tf.int32, shape=[self.batch_size])
            self.train_labels = tf.placeholder(tf.int32, shape=[self.batch_size, 1])

            # Variables.
            embeddings = tf.Variable(
                tf.random_uniform([self.vocabulary_size, self.embedding_size], -1.0, 1.0))
            softmax_weights = tf.Variable(
                tf.truncated_normal([self.vocabulary_size, self.embedding_size],
                                    stddev=1.0 / math.sqrt(self.embedding_size)))
            softmax_biases = tf.Variable(tf.zeros([self.vocabulary_size]))

            # Model.
            # Look up embeddings for inputs.
            embed = tf.nn.embedding_lookup(embeddings, self.train_dataset)

            # Compute the softmax loss, using a sample of the negative labels each time.
            self.loss = tf.reduce_mean(
                tf.nn.sampled_softmax_loss(weights=softmax_weights, biases=softmax_biases, inputs=embed,
                                           labels=self.train_labels, num_sampled=self.num_sampled,
                                           num_classes=self.vocabulary_size))

            # Optimizer.
            # Note: The optimizer will optimize the softmax_weights AND the embeddings.
            # This is because the embeddings are defined as a variable quantity and the
            # optimizer's `minimize` method will by default modify all variable quantities
            # that contribute to the tensor it is passed.
            # See docs on `tf.train.Optimizer.minimize()` for more details.
            self.optimizer = tf.train.AdagradOptimizer(1.0).minimize(self.loss)

            # Compute the similarity between minibatch examples and all embeddings.
            # We use the cosine distance:
            norm = tf.sqrt(tf.reduce_sum(tf.square(embeddings), 1, keep_dims=True))
            self.normalized_embeddings = embeddings / norm


    def train(self, words, num_steps=100001):
        data, _, dictionary, reverse_dictionary = build_dataset(words, vocabulary_size=self.vocabulary_size)
        self.dictionary_ = dictionary
        self.reverse_dictionary_ = reverse_dictionary
        # used to keep track of where we're generating batch data
        self.data_index = 0


        with tf.Session(graph=self.graph) as session:
            tf.global_variables_initializer().run()
            print('Initialized')
            average_loss = 0
            for step in range(num_steps):
                batch_data, batch_labels = self._generate_batch(data)
                feed_dict = {self.train_dataset: batch_data, self.train_labels: batch_labels}
                _, l = session.run([self.optimizer, self.loss], feed_dict=feed_dict)
                average_loss += l
                if step % 2000 == 0:
                    if step > 0:
                        average_loss = average_loss / 2000
                    # The average loss is an estimate of the loss over the last 2000 batches.
                    print('Average loss at step %d: %f' % (step, average_loss))
                    average_loss = 0
                # note that this is expensive (~20% slowdown if computed every 500 steps)

            self.embeddings_ = self.normalized_embeddings.eval()
