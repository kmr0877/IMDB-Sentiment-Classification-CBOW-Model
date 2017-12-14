# IMDB-Sentiment-Classification

We will develop a classifier able to detect the sentiment of movie reviews. Sentiment classification is an active area of research. Aside from improving performance of systems like Siri and Cortana, sentiment analysis is very actively utilized in the finance industry, where sentiment is required for automated trading on news snippits and press releases. 

## DataSet

The training dataset contains a series of movie reviews scraped from the IMBD website. There are no more than 30 reviews for any one specific movie. You have been provided with a tarball (reviews.tar.gz) that contains two folders; "pos" and "neg". It contains the unchanged reviews in plain text form. Each folder contains 12500 positive and negative reviews respectively. Each review is contained in the first line of its associated text file, with no line breaks.
We will be extracting these files, and load them into a datastructure so than we can feed them into TensorFlow. There is also need for some level of preprocessing on this text prior to feeding it into our model. Because the glove embeddings are all in lowercase, we should convert all reviews to lowercase, and also strip punctuation. We may want to do additional preprocessing by stripping out unessesary words etc. 

For the purposes of reducing training time, we limit every review fed into the classifier at 40 words. This should occur after preprocessing. The model will only accept input sequences of length 40. If a review is not 40 words it should be 0-padded. Some reviews are much longer than 40 words, but for this project we will assume the sentiment can be obtained from the first 40.
