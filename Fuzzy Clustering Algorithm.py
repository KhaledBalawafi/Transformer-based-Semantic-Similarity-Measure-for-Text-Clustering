import numpy as np
import skfuzzy as fuzz
import matplotlib.pyplot as plt
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn import metrics


#bert_model = SentenceTransformer('bert-base-nli-mean-tokens')
bert_model = SentenceTransformer('paraphrase-distilroberta-base-v1')


all_sentences = []
labels_true = []
all_labels_pred = []
f = open("Clustering Datasets/searchsnippets dataset.txt")
l = open("Clustering Datasets/searchsnippets dataset labels.txt")

all_sentences = f.readlines()
labels_true_file = l.read()

bert_embeddings = bert_model.encode(all_sentences)

matrix_bert = cosine_similarity(bert_embeddings)

# Fuzzy C-Means clustering
n_clusters = 8
cntr, membership_probs, _, _, _, _, _ = fuzz.cluster.cmeans(matrix_bert.T, n_clusters, 2, error=0.005, maxiter=1000, init=None)

# Assign each sample to the cluster with the highest membership probability
yhat = membership_probs.argmax(axis=0)

for i in yhat:
    all_labels_pred.append(i)

for i in labels_true_file:
    if i != '\n':
        labels_true.append(i)

all_labels_true = [int(i) for i in labels_true]

# Print evaluation metrics
print('Rand index score (accuracy) = ', metrics.rand_score(all_labels_true, all_labels_pred))
print('Fowlkes-Mallows index FMI score = ', metrics.fowlkes_mallows_score(all_labels_true, all_labels_pred))
print('V-measure score (NMI) = ', metrics.v_measure_score(all_labels_true, all_labels_pred))
print(f'Silhouette Score: {metrics.silhouette_score(matrix_bert, yhat)}')

# Create a scatter plot for fuzzy clustering
colors = ['r', 'g', 'b', 'y']  # You can add more colors if you expect more clusters
for i in range(n_clusters):
    row_ix = np.where(yhat == i)
    plt.scatter(matrix_bert[row_ix, 0], matrix_bert[row_ix, 1], label=f'Cluster {i}', c=colors[i])

plt.legend()
plt.xlabel('Feature 1')
plt.ylabel('Feature 2')
plt.title('Fuzzy Clustering')
plt.show()


# Create a scatter plot for Silhouette scores
silhouette_scores = metrics.silhouette_samples(matrix_bert, yhat)
plt.scatter(range(len(silhouette_scores)), silhouette_scores, c=yhat, cmap='viridis')
plt.colorbar(label='Cluster')
plt.axhline(y=np.mean(silhouette_scores), color='red', linestyle='--', label='Average Silhouette Score')
plt.xlabel('Data Points')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Scores for Each Data Point')
plt.legend()
plt.show()
