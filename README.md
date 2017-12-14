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

CBOW vs Skip-gram

Input-Output

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
