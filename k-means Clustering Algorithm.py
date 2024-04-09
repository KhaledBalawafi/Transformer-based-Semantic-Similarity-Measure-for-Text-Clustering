# k-means clustering
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
from sklearn import metrics
from matplotlib import pyplot
from yellowbrick.cluster import SilhouetteVisualizer
from yellowbrick.cluster import KElbowVisualizer
from sklearn.metrics import silhouette_score
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
#import tensorflow_hub as hub
#import numpy as np
#import nltk
#nltk.download('punkt')
#from models import InferSent
#import torch



# Load the sentence transformer models
bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
#bert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')

#RoBERTa_model = SentenceTransformer('roberta-large-nli-stsb-mean-tokens')
#siamese_model = SentenceTransformer('msmarco-distilbert-base-v2')
#quickthought_model = SentenceTransformer('quora-distilbert-base')
#USE_model = hub.load("https://tfhub.dev/google/universal-sentence-encoder-large/5")
#V = 2
#MODEL_PATH = 'encoder/infersent%s.pkl' % V
#params_model = {'bsize': 64, 'word_emb_dim': 300, 'enc_lstm_dim': 2048,
#                'pool_type': 'max', 'dpout_model': 0.0, 'version': V}
#model = InferSent(params_model)
#model.load_state_dict(torch.load(MODEL_PATH))
#W2V_PATH = 'encoder/glove.840B.300d.txt'
#model.set_w2v_path(W2V_PATH)

print ("\n Load the sentence transformer models done\n")


all_sentences = []
labels_true = []
all_labels_pred  = []
f = open("Clustering Datasets/ag-news-dataset.txt") 
l = open("Clustering Datasets/ag-news-dataset labels.txt")
#f = open("Clustering Datasets/yahoo-answers dataset.txt", 'r', encoding='utf-8', errors='ignore') 
#l = open("Clustering Datasets/yahoo-answers dataset labels.txt")


print ("\n Load the dataset done\n")

all_sentences = f.readlines()
labels_true_file = l.read()


# Compute the sentence embeddings for each model
bert_embeddings = bert_model.encode(all_sentences)
#RoBERTa_embeddings = RoBERTa_model.encode(all_sentences)
#siamese_embeddings = siamese_model.encode(all_sentences)
#quickthought_embeddings = quickthought_model.encode(all_sentences)
#USE_embeddings = USE_model(all_sentences)
#model.build_vocab(all_sentences, tokenize=True)
#Infersent_embeddings = model.encode(all_sentences)

print ("\n Load the sentence embedding models done\n")

# Compute the cosine similarity matrices for each model
matrix_bert = cosine_similarity(bert_embeddings)
#matrix_RoBERTa = cosine_similarity(RoBERTa_embeddings)
#matrix_siamese = cosine_similarity(siamese_embeddings)
#matrix_quickthougth = cosine_similarity(quickthought_embeddings)
#matrix_USE = np.inner(USE_embeddings, USE_embeddings)
#matrix_Infersent = cosine_similarity(Infersent_embeddings)

print ("\n Load the similarity matrix done\n")

#------------------------------BERT--------------------------------

# define the model
model = KMeans(n_clusters=10)
# fit the model
model.fit(matrix_bert)
# assign a cluster to each example
yhat = model.predict(matrix_bert)
# retrieve unique clusters
clusters = unique(yhat)

for i in yhat:
    all_labels_pred.append(i)
for i in labels_true_file:
    if i != '\n':
        labels_true.append(i)

all_labels_true = [int(i) for i in labels_true]


print ('Rand index score (accuracy) = ', metrics.rand_score(all_labels_true, all_labels_pred))
print ('Fowlkes-Mallows index FMI score = ', metrics.fowlkes_mallows_score(all_labels_true, all_labels_pred))
print ('V-measure score (NMI) = ', metrics.v_measure_score(all_labels_true, all_labels_pred))
print(f'Silhouette Score: {silhouette_score(matrix_bert, yhat)}')

# create scatter plot for Silhouette measure
visualizer = SilhouetteVisualizer(model, colors='yellowbrick')
visualizer.fit(matrix_bert)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

# create scatter plot for Elbow measure
visualizerA = KElbowVisualizer(model)
visualizerA.fit(matrix_bert)        # Fit the data to the visualizer
visualizerA.show()        # Finalize and render the figure



# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(matrix_bert[row_ix, 0], matrix_bert[row_ix, 1], label = cluster)
    
pyplot.show()

