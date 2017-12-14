import numpy as np
import matplotlib
import glob
import re
from collections import Counter
from string import punctuation
# if you get the error: "TypeError: 'figure' is an unknown keyword argument"
# uncomment the line below:
# matplotlib.use('Qt4Agg')

try:
    # pylint: disable=g-import-not-at-top
    from sklearn.manifold import TSNE
    import matplotlib.pyplot as plt
except ImportError as e:
    print(e)
    print('Please install sklearn, matplotlib, and scipy to show embeddings.')
    exit()

def plot_with_labels(low_dim_embs, labels, filename='tsne_embeddings.png'):
    assert low_dim_embs.shape[0] >= len(labels), 'More labels than embeddings'

    plt.figure(figsize=(18, 18))  # in inches
    for i, label in enumerate(labels):
        x, y = low_dim_embs[i, :]
        plt.scatter(x, y)
        plt.annotate(label,
                     xy=(x, y),
                     xytext=(5, 2),
                     textcoords='offset points',
                     ha='right',
                     va='bottom')

    plt.savefig(filename)
    print("plots saved in {0}".format(filename))
vocabulary_size = 1000


def read_and_clean_data(path):

    with open(path,"r") as o:
        text = o.read()
        punc_rem = re.sub(r'[^\w\s]', '', text)
        #punc_rem = text.translate(None, punctuation)
        lower_words = map(lambda x: x.lower(),punc_rem.split())

    return lower_words
if __name__ == "__main__":
    # Step 6: Visualize the embeddings.
    corpus = []
    count = []
    words = []
    folders = ["neg", "pos"]
    for folder in folders:
        for path in glob.glob(folder + "/*.txt"):

            words += read_and_clean_data(path)
            if len(set(words)) > vocabulary_size :
                break
        else:
            continue
        break

    count = Counter(words)
    unique_words = sorted(count.keys())
    idxs = range(len(count.keys()))
    data = "the first that is the first the and do not bad and not good to the".split()
    reverse_dictionary = dict(zip(unique_words, idxs))
    dictionary = dict(zip(idxs, unique_words))
    # reverse_dictionary = np.load("Idx2Word.npy").item()
    embeddings = np.load("CBOW_Embeddings.npy")
    tsne = TSNE(perplexity=30, n_components=2, init='pca', n_iter=5000, method='exact')
    plot_only = 500
    low_dim_embs = tsne.fit_transform(embeddings[:plot_only, :])
    labels = [dictionary[i] for i in range(plot_only)]
    plot_with_labels(low_dim_embs, labels)
    plt.show()