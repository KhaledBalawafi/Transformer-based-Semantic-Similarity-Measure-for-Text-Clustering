# Transformer-based-Semantic-Similarity-Measure-for-Text-Clustering
Replicating Experimental Results and Methods
This repository contains a selection of code used for our journal paper titled "Experimental Study on Short-text Clustering Using Transformer-based Semantic Similarity Measure." Below, we outline the available resources and instructions for replicating our results.

Paper Abstract
Sentence clustering is essential in text-processing activities, particularly for measuring semantic similarity between sentences. In our paper, we explore the use of a sentence similarity measure based on embedding representations derived from pre-trained models. We evaluate this measure with three text clustering methods—partitional, hierarchical, and fuzzy clustering—on standard textual datasets. Our study showcases the effectiveness of embedding-based similarity measures in enhancing clustering and summarization tasks.

Available Code
Within this repository's “codes” folder, you will find the following scripts:
- k-means Clustering Algorithm: Python script implementing the k-means clustering method.
- Agglomerative Clustering Algorithm: Python script for agglomerative clustering.
- Fuzzy Clustering Algorithm: Python script for fuzzy clustering.
- AgglomerativeSummarization: Script demonstrating text summarization using agglomerative clustering.
- Summary-Plots: Script generating plots summarizing the results.

Replication Instructions
To replicate our experiments and results, follow these steps:
1. Clone the Repository: Start by cloning this repository to your local machine.
2. Data Setup: Ensure you have access to the standard textual datasets used in our study. If these datasets are publicly available, provide instructions or links for accessing them.
3. Install Dependencies: Install the required Python dependencies. You may use “pip” for this purpose.
   pip install -r requirements.txt
4. Run Scripts: Execute the relevant scripts to reproduce specific experiments.
   python k-means_Clustering_Algorithm.py
   python Agglomerative_Clustering_Algorithm.py
   python Fuzzy_Clustering_Algorithm.py
   python AgglomerativeSummarization.py
   python Summary-Plots.py
   
Note on Code Availability
While the provided scripts cover key aspects of our methods and experiments, the complete codebase used for our paper is not included in this repository. For access to the full implementation and preprocessed data, please reach out to the authors for further details.

For any inquiries or assistance regarding our study, please contact Khaled Abdalgader at Khaled.balawafi@aurak.ac.ae.

We hope these resources assist in replicating and understanding our experimental approach. Thank you for your interest in our work!
