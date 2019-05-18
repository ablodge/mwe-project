# -*- coding: utf-8 -*-

import re, h5py
import numpy as np
from collections import Counter
from keras.utils import to_categorical
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding
from gensim.models import KeyedVectors
from keras.preprocessing.text import Tokenizer

from corpus_reader import *

'''
Called in main.py, used in train_test.py

Converts CUPT style input to IBOo,
    converts them to tensor encoding,
    loads pretrained embeddings,
    adds adjacency matrix.

Attributes:
    d.max_length        : max sentence length
    d.input_dim         : dimension of embedding
    d.n_poses           : number of pos tags
    d.n_classes         : number of classes
    d.depAdjacency_gcn  : boolean (later updated in train_test.py)
    d.position_embed    : boolean (get_position_encoding)
    d.w2idx             : vocab
    d.X_train_enc       : list(list(word index))
    d.X_test_enc        : list(list(word index))
    d.y_train_enc       : list(list(list(label one hot)))
    d.y_test_enc        : list(list(list(label one hot)))
    d.train_weights     : from load elmo
    d.test_weights      : from load elmo
    d.embedding_layer :
        self.embedding_layer = Embedding(embedding_matrix.shape[0],
                                             embedding_matrix.shape[1],
                                             weights=[embedding_matrix],
                                             trainable=False,
                                             name='embed_layer')

'''


class Data(object):
    """A preprocessor that prepares the data to be trained by the model

    Attributes:
        * lang: name of the language based on the language codes used in Parseme
        * word2vec_dir: path to the pre-trained word2vec
        * elmo_dir: path to the pretrained elmo
        * model_name: name of the learning model
        * testORdev: whether we are getting results on test or development set
    """

    def __init__(self, lang, testORdev, glove_file, elmo_file, model_name, depAdjacency_gcn=0, position_embed=False):
        self.lang = lang
        self.glove_file = glove_file
        self.elmo_file = elmo_file
        self.input_dim = 0
        self.model_name = model_name
        self.testORdev = testORdev
        self.depAdjacency_gcn = depAdjacency_gcn
        self.position_embed = position_embed
        self.input_dim = 100

    def encode(self, sents):
        """integer encode the sentences
        """
        t = Tokenizer(filters='\t\n', lower=False)
        t.fit_on_texts([" ".join(sent) for sent in sents])
        return t.word_index

    def word_shape(self, word):
        """incorporate word shape information

            * 1 initial captial
            * 2 all char capital
            * 3 has #
            * 4 has @
            * 5 is URL
            * 6 is number
            * 7 has number
        """
        vector = np.zeros(7)
        if word[0].isupper():
            vector[0] = 1
        if word.isupper():
            vector[1] = 1
        if word[0] == '#':
            vector[2] = 1
        if word[0] == "@":
            vector[3] = 1
        if "http" in word[0:6]:
            vector[4] = 1
        if word.isdigit():
            vector[5] = 1
        if any(char.isdigit() for char in word):
            vector[6] = 1
        return vector

    def load_data(self, path):
        """reading train and test
        """
        print("Reading the corpus .......")
        c = Corpus_reader(path + self.lang + "/")
        # train = pickle.load(open('../{}/{}.pkl'.format(self.lang, train), 'rb'))
        train = c.read(c.train_sents)
        X_train = [
            [x[0].replace('.', "$period$").replace("\\", "$backslash$").replace("/", "$backslash$") for x in elem] for
            elem in train]
        y_train = [[x[5] for x in elem] for elem in train]
        pos_train = [[x[2] for x in elem] for elem in train]
        self.dep_train = [[x[3] for x in elem] for elem in train]
        if self.lang != "EN" and self.testORdev == "TEST":
            dev = c.read(c.dev_sents)
            # dev_file = pickle.load(open('../{}/{}.pkl'.format(self.lang, self.dev), 'rb'))
            X_train = X_train + [
                [x[0].replace('.', "$period$").replace("\\", "$backslash$").replace("/", "$backslash$") for x in elem]
                for elem in dev]
            y_train = y_train + [[x[5] for x in elem] for elem in dev]
            pos_train = pos_train + [[x[2] for x in elem] for elem in dev]
            self.dep_train = self.dep_train + [[x[3] for x in elem] for elem in dev]
        if self.testORdev == "TEST":
            test = c.read(c.test_sents)
        elif self.testORdev == "DEV":
            test = c.read(c.dev_sents)
        elif self.testORdev == "CROSS_VAL":
            test = []
        else:
            print("ERROR: please specify if it is test or development!")

        # test = pickle.load(open('../{}/{}.pkl'.format(self.lang, self.test), 'rb'))
        X_test = [[x[0].replace('.', "$period$").replace("\\", "$backslash$").replace("/", "$backslash$") for x in elem]
                  for elem in test]
        y_test = [[x[5] for x in elem] for elem in test]
        pos_test = [[x[2] for x in elem] for elem in test]
        self.dep_test = [[x[3] for x in elem] for elem in test]
        ### ### ###
        self.max_length = len(max(X_train + X_test, key=len))
        print("max sentence length:", self.max_length)

        ######################################

        words = list(set([elem for sublist in X_train + X_test for elem in sublist]))
        self.vocab_size = len(words) + 2  # because of <UNK> and <PAD> pseudo words
        self.n_classes = len(
            set([elem for sublist in (y_train + y_test) for elem in sublist])) + 1  # add 1 because of zero padding
        self.n_poses = len(set([elem for sublist in (pos_train + pos_test) for elem in sublist])) + 1
        print("number of POS: ", self.n_poses)

        # assign a unique integer to each word/label
        self.w2idx = {word: i + 1 for (i, word) in enumerate(words)}
        # w2idx = encode(X_train+X_test)
        self.l2idx = self.encode(y_train + y_test)
        self.pos2idx = self.encode(pos_train + pos_test)

        # encode() maps each word to a unique index, starting from 1. We additionally incerement all the
        # values by 1, so that we can save space for 0 and 1 to be assigned to <PAD> and <UNK> later
        self.w2idx = Counter(self.w2idx)
        self.w2idx.update(self.w2idx.keys())
        self.w2idx = dict(
            self.w2idx)  # convert back to regular dict (to avoid erroneously assigning 0 to unknown words)

        self.w2idx['<PAD>'] = 0
        self.w2idx['<UNK>'] = 1

        # on the label side we only have the <PADLABEL> to add
        self.l2idx['<PADLABEL>'] = 0
        self.pos2idx['<PADPOS>'] = 0

        # keep the reverse to be able to decode back
        self.idx2w = {v: k for k, v in self.w2idx.items()}
        self.idx2l = {v: k for k, v in self.l2idx.items()}
        self.idx2pos = {v: k for k, v in self.pos2idx.items()}

        self.X_train_enc = [[self.w2idx[w] for w in sent] for sent in X_train]
        self.X_test_enc = [[self.w2idx[w] for w in sent] for sent in X_test]

        self.y_train_enc = [[self.l2idx[l] for l in labels] for labels in y_train]
        self.y_test_enc = [[self.l2idx[l] for l in labels] for labels in y_test]

        self.pos_train_enc = [[self.pos2idx[p] for p in poses] for poses in pos_train]
        self.pos_test_enc = [[self.pos2idx[p] for p in poses] for poses in pos_test]

        # zero-pad all the sequences

        self.X_train_enc = pad_sequences(self.X_train_enc, maxlen=self.max_length, padding='post')
        self.X_test_enc = pad_sequences(self.X_test_enc, maxlen=self.max_length, padding='post')

        self.y_train_enc = pad_sequences(self.y_train_enc, maxlen=self.max_length, padding='post')
        self.y_test_enc = pad_sequences(self.y_test_enc, maxlen=self.max_length, padding='post')

        self.pos_train_enc = pad_sequences(self.pos_train_enc, maxlen=self.max_length, padding='post')
        self.pos_test_enc = pad_sequences(self.pos_test_enc, maxlen=self.max_length, padding='post')

        # one-hot encode the labels
        self.idx = np.array(list(self.idx2l.keys()))
        self.vec = to_categorical(self.idx)
        self.one_hot = dict(zip(self.idx, self.vec))
        self.inv_one_hot = {tuple(v): k for k, v in self.one_hot.items()}  # keep the inverse dict

        self.y_train_enc = np.array([[self.one_hot[l] for l in labels] for labels in self.y_train_enc])
        self.y_test_enc = np.array([[self.one_hot[l] for l in labels] for labels in self.y_test_enc])

        # one-hot encode the pos tags
        self.idx = np.array(list(self.idx2pos.keys()))
        self.vec = to_categorical(self.idx)
        self.pos_one_hot = dict(zip(self.idx, self.vec))
        self.inv_pos_one_hot = {tuple(v): k for k, v in self.pos_one_hot.items()}  # keep the inverse dict

        self.pos_train_enc = np.array([[self.pos_one_hot[p] for p in poses] for poses in self.pos_train_enc])
        self.pos_test_enc = np.array([[self.pos_one_hot[p] for p in poses] for poses in self.pos_test_enc])
        print("train pos array shape", self.pos_train_enc.shape)  # pos information is not necessarily used by the model

        if self.elmo_file:
            # load_elmo()
            self.input_dim = 1024
            self.embedding_layer = None
        elif self.glove_file:
            self.load_glove()
        else:
            self.embedding_layer = Embedding(len(self.w2idx)+1,
                                             self.input_dim,
                                             trainable=True,
                                             name='embed_layer')

        if self.depAdjacency_gcn:
            train_adjacency = self.load_adjacency(self.dep_train, 1)
            test_adjacency = self.load_adjacency(self.dep_test, 1)

            self.train_adjacency_matrices = [train_adjacency]
            self.test_adjacency_matrices = [test_adjacency]

            print("adjacency matrices size", len(self.train_adjacency_matrices))

            # self.input_dim = len(self.train_weights[0][0])

        # if self.position_embed:  # I think this is not correct, the position embedding was not correct from the beginning.
            # self.train_weights = np.add(self.train_weights, self.get_pos_encoding(1024))
            # print('size of the position encoded embedding: ', self.train_weights.shape)
        #	lst = np.add(lst, self.get_pos_encoding(1024))
        #	print('size of the position encoded embedding: ', lst.shape)

    def load_glove(self):
        if not self.glove_file:
            pass  # do nothing if there is no path to a pre-trained embedding avialable
        else:
            print("loading glove ...")
            embeddings_index = {}
            f = open(self.glove_file, encoding='utf8')
            for line in f:
                values = line.split()
                word = values[0]
                coefs = np.asarray(values[1:], dtype='float32')
                embeddings_index[word] = coefs
            f.close()

            print('Found %s word vectors.' % len(embeddings_index))

            embedding_matrix = np.zeros((len(self.w2idx), self.input_dim))
            for word, i in self.w2idx.items():
                embedding_vector = embeddings_index.get(word)
                if embedding_vector is not None:
                    # words not found in embedding index will be all-zeros.
                    embedding_matrix[i] = embedding_vector

            self.embedding_layer = Embedding(embedding_matrix.shape[0],
                                             embedding_matrix.shape[1],
                                             weights=[embedding_matrix],
                                             trainable=False,
                                             name='glove_layer')

    def load_elmo(self, X):  # the aim is to create a numpy array of shape (sent_num, max_sent_size, 1024)
        if not self.elmo_file:
            pass  # do nothing if there is no path to a pre-trained elmo avialable
        else:
            # self.embedding_layer = ElmoEmbeddingLayer()
            print('ELMO Loaded ...')
            # return np.array(lst, dtype=np.float32)

    # def load_headVectors(self, X, dep):
    #
    #     filename = self.elmo_dir + '/ELMo_{}'.format(self.lang)
    #     elmo_dict = h5py.File(filename, 'r')
    #     lst = []
    #     sentIndx = 0
    #     for sent_toks, dep_toks in zip(X, dep):
    #         sent = "\t".join(sent_toks)
    #         if sent in elmo_dict:
    #             item = list(elmo_dict[sent])
    #         else:
    #             item = list(np.zeros((len(sent_toks), 1024)))
    #         min_lim = len(item)
    #         sent_deps = []
    #         for i in range(0, len(sent_toks)):
    #             if int(dep[sentIndx][i]) > min_lim and min_lim != 0:
    #                 print("error ", str(dep[sentIndx][i]), "greater than the sent length ", str(min_lim))
    #             if int(dep[sentIndx][i]) - 1:
    #                 sent_deps.append(item[int(dep_toks[i]) - 1])
    #             else:  # the word is the root in the sent
    #                 # item[i] = list(item[i]).extend([0]*1024)
    #                 sent_deps.append(item[i])
    #
    #         for i in range(min_lim, self.max_length):
    #             sent_deps.append([0] * 1024)
    #
    #         lst.append(sent_deps)
    #         sentIndx += 1
    #     print("dep head vectors shape: ", np.array(lst).shape)
    #     return np.array(lst)

    def load_adjacency(self, dep, direction):

        if direction == 1:
            dep_adjacency = [self.adjacencyHead2Dep(d) for d in dep]
        elif direction == 0:
            dep_adjacency = [self.adjacencyDep2Head(d) for d in dep]
        elif direction == 3:
            dep_adjacency = [self.adjacencySelf(d) for d in dep]

        return np.array(dep_adjacency)

    def adjacencyDep2Head(self, sentDep):
        adjacencyMatrix = np.zeros((self.max_length, self.max_length), dtype=np.int)
        for i in range(len(sentDep)):
            if sentDep[i] != 0:
                adjacencyMatrix[i][sentDep[i] - 1] = 1
            # adjacencyMatrix[sentDep[i]-1][i] = -1
        return adjacencyMatrix

    def adjacencyHead2Dep(self, sentDep):
        adjacencyMatrix = np.zeros((self.max_length, self.max_length), dtype=np.int)
        for i in range(len(sentDep)):
            if sentDep[i] != 0:
                # adjacencyMatrix[i][sentDep[i]-1] = 1
                adjacencyMatrix[sentDep[i] - 1][i] = 1
        return adjacencyMatrix

    def adjacencySelf(self, sentDep):
        adjacencyMatrix = np.zeros((self.max_length, self.max_length), dtype=np.int)
        for i in range(len(sentDep)):
            adjacencyMatrix[i][i] = 1
        return adjacencyMatrix

    def get_pos_encoding(self, d_emb):
        """outputs a position encoding matrix"""
        pos_enc = np.array([
            [pos / np.power(10000, 2 * (j // 2) / d_emb) for j in range(d_emb)]
            if pos != 0 else np.zeros(d_emb)
            for pos in range(self.max_length)
        ])
        pos_enc[1:, 0::2] = np.sin(pos_enc[1:, 0::2])  # dim 2i
        pos_enc[1:, 1::2] = np.cos(pos_enc[1:, 1::2])  # dim 2i+1
        return pos_enc


# # taken from
# class ElmoEmbeddingLayer(Layer):
#
#     def __init__(self, **kwargs):
#         self.dimensions = 100
#         self.trainable = False
#         super(ElmoEmbeddingLayer, self).__init__(**kwargs)
#
#     def build(self, input_shape):
#         self.elmo = hub.Module('https://tfhub.dev/google/elmo/2', trainable=self.trainable,
#                                name="{}_module".format(self.name))
#         self.trainable_weights += K.tf.trainable_variables(scope="^{}_module/.*".format(self.name))
#         super(ElmoEmbeddingLayer, self).build(input_shape)
#
#     def call(self, x, mask=None):
#         result = self.elmo(K.squeeze(K.cast(x, tf.string), axis=1),
#                            as_dict=True,
#                            signature='default',
#                            )['default']
#         return result
#
#     def compute_mask(self, inputs, mask=None):
#         return K.not_equal(inputs, '<PAD>')
#
#     def compute_output_shape(self, input_shape):
#         return (input_shape[0], self.dimensions)
