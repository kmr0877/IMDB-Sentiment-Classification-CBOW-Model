
#written by kmr0877 for comp9444
import tensorflow as tf
import numpy as np
from string import punctuation
import glob #this will be useful when reading reviews from file
import re
import tarfile


batch_size = 50

def load_data(glove_dict):
    """
    Take reviews from text files, vectorize them, and load them into a
    numpy array. Any preprocessing of the reviews should occur here. The first
    12500 reviews in the array should be the positive reviews, the 2nd 12500
    reviews should be the negative reviews.
    RETURN: numpy array of data with each row being a review in vectorized
    form"""

    def read_and_clean_data(path):

        with open(path, "r") as o:
            text = o.read()
            punc_rem = re.sub(r'[^\w\s]', '', text)
            vec_words = []
            words = punc_rem.split()
            if len(words)> 40:
                words = words[:40]
            else:
                padding = np.zeros(40 - len(words))
                words += list(padding)
            for word in words:
                try:
                    idx = glove_dict[word.lower()]
                except:
                    idx = 0
                vec_words.append(idx)

        return vec_words


    data = []
    folders = ["pos", "neg"]
    for folder in folders:
        for path in glob.glob(folder + "/*.txt"):
            vectorized_sent = read_and_clean_data(path)
            data.append(vectorized_sent)
    return data


def load_glove_embeddings():
    """
    Load the glove embeddings into a array and a dictionary with words as
    keys and their associated index as the value. Assumes the glove
    embeddings are located in the same directory and named "glove.6B.50d.txt"
    RETURN: embeddings: the array containing word vectors
            word_index_dict: a dictionary matching a word in string form to
            its index in the embeddings array. e.g. {"apple": 119"}
    """
    with open("glove.6B.50d.txt",'r') as o:
        embeddings = o.readlines()
    words_list = map(lambda x: x.split(" ", 1), embeddings)
    a, b = zip(*words_list)
    word_index_dict = dict(zip(a, range(len(a))))
    # if you are running on the CSE machines, you can load the glove data from here
    # data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")
    glove_array = np.array(list(map(lambda x: x.split(), b)))
    glove_array = glove_array.astype(np.float32)
    return glove_array, word_index_dict


def define_graph(glove_embeddings_arr):
    """
    Define the tensorflow graph that forms your model. You must use at least
    one recurrent unit. The input placeholder should be of size [batch_size,
    40] as we are restricting each review to it's first 40 words. The
    following naming convention must be used:
        Input placeholder: name="input_data"
        labels placeholder: name="labels"
        accuracy tensor: name="accuracy"
        loss tensor: name="loss"

    RETURN: input placeholder, labels placeholder, optimizer, accuracy and loss
    tensors"""
    max_document_length = 40
    sequence_length = max_document_length
    rnn_data_classes = 2
    rnn_cell_size = 100
    rnn_data_embedding_size = 50
    rnn_data_vec_size = rnn_data_embedding_size
    rnn_lstm_forget_bias = 1.0
    rnn_dropout_keep_prob = 0.5
    rnn_learning_rate = 0.001
    vocab_size = len(glove_embeddings_arr)
    with tf.device("/cpu:0"):
        # Placeholders for input, output and dropout
        labels = tf.placeholder(tf.int32, [None, rnn_data_classes], name="labels")
    #     print labels.shape
        input_data = tf.placeholder(tf.int32, [None, max_document_length],
                                 name="input_data")  # Place holder size format [rows, columns].
    #     print input_data.shape
    #       # one column.
    #
    # #     # Define weights and biases for linear activation, using rnn inner loop last output
        weights = tf.Variable(tf.random_normal([rnn_cell_size, rnn_data_classes]))
        biases = tf.Variable(tf.random_normal([rnn_data_classes]))
        dropout_keep_prob = tf.placeholder_with_default(1.0, [])

        # embedding = tf.get_variable("embedding", [vocab_size, embedding_size])
        # print embedding.shape
        #print(input_data.shape)
        embedded_data = tf.nn.embedding_lookup(glove_embeddings_arr, input_data)
        #print(embedded_data.shape)
        embedded_data_dropout = tf.nn.dropout(embedded_data, rnn_dropout_keep_prob)
        #print(embedded_data_dropout.shape)
        # add LSTM cell and dropout nodes
    rnn_lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(rnn_cell_size, forget_bias=rnn_lstm_forget_bias)
    rnn_lstm_cell = tf.nn.rnn_cell.DropoutWrapper(rnn_lstm_cell, output_keep_prob=dropout_keep_prob)


    #print("RNN rnn_data_X: ", embedded_data_dropout)

    rnn_data_X = embedded_data_dropout
    # print(rnn_data_X.shape_)
    # Permuting batch_size and sequence_length
    rnn_data_X1 = tf.transpose(rnn_data_X, [1, 0, 2])
    # print rnn_data_X1.shape
    # print ("RNN After transpose rnn_data_X: ", rnn_data_X)
    # Reshaping to (sequence_length * batch_size, rnn_data_vec_size)
    rnn_data_X2 = tf.reshape(rnn_data_X1, [-1, rnn_data_vec_size])
    # print rnn_data_X2.shape
    # print ("RNN After reshape rnn_data_X: ", rnn_data_X)
    # Split to get a list of 'sequence_length' tensors of shape (batch_size, rnn_data_vec_size)
    rnn_data_X_splited = tf.split(rnn_data_X2, sequence_length, 0)
    # print len(rnn_data_X_splited),rnn_data_X_splited[0].shape
    # print ("RNN After split len(rnn_data_X): ", len(rnn_data_X), rnn_data_X[0])

    # # Get lstm cell output
    outputs, states = tf.nn.static_rnn(rnn_lstm_cell, rnn_data_X_splited, dtype=tf.float32)
    # outputs = tf.stack(outputs)
    # print outputs[-1].shape
    logits = tf.matmul(outputs[-1], weights) + biases
    # Define loss and optimizer
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=labels))
    optimizer = tf.train.AdamOptimizer(learning_rate=rnn_learning_rate).minimize(loss)
    #
    # # Evaluate model
    correct_prediction = tf.equal(tf.argmax(logits, 1), tf.argmax(labels, 1))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32),name="accuracy")
    #print(accuracy.name)
    return input_data,labels, dropout_keep_prob, optimizer, accuracy, loss
