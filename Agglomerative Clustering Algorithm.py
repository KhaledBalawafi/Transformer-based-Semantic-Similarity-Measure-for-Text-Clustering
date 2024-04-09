# agglomerative clustering
from numpy import unique
from numpy import where
from sklearn.cluster import AgglomerativeClustering
from matplotlib import pyplot
from yellowbrick.cluster import SilhouetteVisualizerSC
from sklearn.metrics import silhouette_score
from sklearn import metrics
from yellowbrick.cluster import KElbowVisualizer
from sentence_transformers import SentenceTransformer

#embedder = SentenceTransformer('bert-base-nli-mean-tokens')
embedder = SentenceTransformer('paraphrase-distilroberta-base-v1')


all_sentences = []
labels_true = []
all_labels_pred  = []
f = open("Clustering Datasets/searchsnippets dataset.txt") 
l = open("Clustering Datasets/searchsnippets dataset labels.txt")

#f = open("Clustering Datasets/yahoo-answers dataset.txt", 'r', encoding='utf-8', errors='ignore') 
#l = open("Clustering Datasets/yahoo-answers dataset labels.txt")
all_sentences = f.readlines()
labels_true_file = l.read()
matrix = embedder.encode(all_sentences)

print ('done')

# define the model
model = AgglomerativeClustering(n_clusters=8)
# fit model and predict clusters
yhat = model.fit_predict(matrix)
# retrieve unique clusters
clusters = unique(yhat)


for i in yhat:
    all_labels_pred.append(i)
for i in labels_true_file:
    if i != '\n':
        labels_true.append(i)

all_labels_true = [int(i) for i in labels_true]

#print (all_labels_true, '\n\n', all_labels_pred)


# Clustering evalutation criteria
#print ('Rand index score (adjusted) ARI = ', metrics.adjusted_rand_score(all_labels_pred, all_labels_true))
#print ('homogeneity score = ', metrics.homogeneity_score(all_labels_true, all_labels_pred))
#print ('completeness score = ', metrics.completeness_score(all_labels_true, all_labels_pred))
#print ('Normalized Mutual Information NMI score = ', metrics.normalized_mutual_info_score(all_labels_true, all_labels_pred))
#print ('Adjusted Mutual Information AMI score = ', metrics.adjusted_mutual_info_score(all_labels_true, all_labels_pred)) 

print ('Rand index score (accuracy) = ', metrics.rand_score(all_labels_true, all_labels_pred))
print ('Fowlkes-Mallows index FMI score = ', metrics.fowlkes_mallows_score(all_labels_true, all_labels_pred))
print ('V-measure score (NMI) = ', metrics.v_measure_score(all_labels_true, all_labels_pred))
print(f'Silhouette Score: {silhouette_score(matrix, yhat)}')

# create scatter plot for Silhouette measure
visualizer = SilhouetteVisualizerSC(model, colors='yellowbrick')
visualizer.fit(matrix)        # Fit the data to the visualizer
visualizer.show()        # Finalize and render the figure

# create scatter plot for Elbow measure
visualizerA = KElbowVisualizer(model)
visualizerA.fit(matrix)        # Fit the data to the visualizer
visualizerA.show()        # Finalize and render the figure




# create scatter plot for samples from each cluster
for cluster in clusters:
	# get row indexes for samples with this cluster
	row_ix = where(yhat == cluster)
	# create scatter of these samples
	pyplot.scatter(matrix[row_ix, 0], matrix[row_ix, 1], label = cluster)
    
# show the plot
#pyplot.colorbar(ticks=range(100))
#pyplot.clim(-0.5, 9.5)
#pyplot.title("TSNE Visualization")
#pyplot.legend()
pyplot.show()

