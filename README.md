# IMDB-Sentiment-Classification

We will develop a classifier able to detect the sentiment of movie reviews. Sentiment classification is an active area of research. Aside from improving performance of systems like Siri and Cortana, sentiment analysis is very actively utilized in the finance industry, where sentiment is required for automated trading on news snippits and press releases. 

## DataSet

The training dataset contains a series of movie reviews scraped from the IMBD website. There are no more than 30 reviews for any one specific movie. You have been provided with a tarball (reviews.tar.gz) that contains two folders; "pos" and "neg". It contains the unchanged reviews in plain text form. Each folder contains 12500 positive and negative reviews respectively. Each review is contained in the first line of its associated text file, with no line breaks.
We will be extracting these files, and load them into a datastructure so than we can feed them into TensorFlow. There is also need for some level of preprocessing on this text prior to feeding it into our model. Because the glove embeddings are all in lowercase, we should convert all reviews to lowercase, and also strip punctuation. We may want to do additional preprocessing by stripping out unessesary words etc. 

For the purposes of reducing training time, we limit every review fed into the classifier at 40 words. This should occur after preprocessing. The model will only accept input sequences of length 40. If a review is not 40 words it should be 0-padded. Some reviews are much longer than 40 words, but for this project we will assume the sentiment can be obtained from the first 40.

## Stage 1: Word Embeddings

Word embeddings have been shown to improve the performance of many NLP models by converting words from character arrays to vectors that contain semantic infomation of the word itself. In this assignment, you will implement a Continuous Bag of Words (CBOW) version of word2vec - one of the fastest and most commonly used embedding algorithms.
A good introduction to word embeddings can be found in the TensorFlow word2vec tutorial. 

The aim of this task is to modify the code here(https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/word2vec/word2vec_basic.py) so that it uses the continuous bag of words (CBOW) model instead of the skip-gram model. This will produce better embeddings, particularly when less data is available. Furthermore, implementing this change will give us a better understanding of both models, and the differences between them.

## _CBOW vs Skip-gram_

## Input-Output

The main difference between the skip-gram and CBOW model, is the way training data is presented.

With the skip-gram model, the input is the word in the middle of the context window, and the target is to predict any context word (word that is skip_window words to the left or the right) for the given word.

With CBOW, the input is all the words in the context besides the middle word, and the target is to predict the middle word, that was omitted from the context window.

For example, given the sentence fragment "the cat sat on the", the following training examples would be used by skip-gram, with parameters skip_window=1, num_skips=2 - in the form: [words in context window]: (input, target)

[the cat sat]: (cat, the), (cat, sat)
[cat sat on]: (sat, cat), (sat, on),
[sat on the]: (on, sat), (on, the)

While for CBOW the input-output pairs are (note that the inputs now contain more than one word):

[the cat sat]: ([the sat], cat),
[cat sat on]: ([cat on], the),
[sat on the]: ([sat the], on)

Of course, as is explained in the tutorial, the words themselves aren't actually used, but rather their (integer) index into the vocabulary (dictionary) for the task.

## CBOW Input: Mean of Context Words Embeddings

In the skip-gram model there is just a single word as the input, and this word's embedding is looked up, and passed to the predictor.

In the CBOW, since there's more than one word in the context we just take the mean (average) of the embeddings for all context words 

## Testing and Implementatio

word2vec_fns.py	- word2vec implementation
word2vec_cbow.py - code to train our word2vec model
imdb_sentiment_data.py - helper functions for loading the sentiment data, used by word2vec_cbow
plot_embeddings.py	- to visualise embeddings

The file ## word2vec_fns.py  contains two functions:

generate_batch(...) which is initially identical to the function in https://github.com/tensorflow/tensorflow/blob/r1.3/tensorflow/examples/tutorials/word2vec/word2vec_basic.py, with just one change, the num_skips parameter has been removed as it is not needed in the CBOW regime.
get_mean_context_embeds(...)

You can run the code that does the embeddings with:

python3 word2vec_cbow.py

If this completes without error, you should see a file called CBOW_Embeddings.npy in the current directory.

Additionally, if you run

python3 plot_embeddings.py

you should be able to see a low dimensional visualisation of the embeddings created with TSNE. Don't worry if you are unable to get the visualisation running.( you may need to use pip3 in order to get matplotlib installed correctly).

## Stage 2: Sentiment Classifier

implementation.py - RNN classifier implementation

train.py	- file that calls implementation.py and trains our sentiment model

You will also need to download the file glove.6B.50d.txt.gz and gunzip it. If you are running on the Lab machines, you could use the copy of glove.6B.50d.txt in the class account by uncommenting this line in implementation.py (and commenting out the line above it)
data = open("/home/cs9444/public_html/17s2/hw2/glove.6B.50d.txt",'r',encoding="utf-8")

I have made use of recurrent network elements . Aside from the fact that this is the type of network this assessment aims to assess, for text classification some recurrency will be important. Consider the review fragment; "I really thought this was a great example of how not to make a movie.". A naive classifier (e.g. a feed forward network trained on word counts) would be unable to correctly identify the sentiment as it depends on the tail end of the review being understood in the context of the "not" negation. Recurrent units allow us to preserve this dependency as we parse the review.

## Visualizing Your Progress

In addition to the output of  train.py, you can view the progress of your models using the tensorboard logging included in that file. To view these logs, run the following command from the src directory:

python3 -m tensorflow.tensorboard --logdir=./tensorboard

Depending on your installation, the following command might also work:

tensorboard --logdir=./tensorboard

open a Web browser and navigate to  http://localhost:6006

you should be able to see a plot of the loss and accuracies in TensorBoard under the "scalars" tab

Make sure you are in the same directory from which train.py is running. A good resource is here(https://www.tensorflow.org/get_started/summaries_and_tensorboard) for more information.

