# Question 4: Clustering – Gaussian Mixture Models, Cluster Ensembles, and Stability Analysis

## Objective
The objective of this task is to apply advanced clustering techniques beyond standard K-Means, focusing on model selection, ensemble methods, and rigorous stability evaluation.
## Dataset
* **Dataset Used:** A real-world dataset (Wholesale Customer) and `optdigits` for external validation.

## Implementation Tasks

### 1. Clustering Algorithms & Model Selection
Implement and tune four distinct clustering algorithms:
* **K-Means:** Determining optimal $k$ using the elbow method, silhouette score, and gap statistic.
* **Gaussian Mixture Model (GMM):** Using BIC/AIC for optimal component selection.
* **DBSCAN:** Tuning the `eps` parameter via a k-distance graph.
* **Agglomerative Clustering:** Utilizing Ward linkage and dendrogram analysis.

### 2. Evaluation Metrics
* **Internal Metrics:** Silhouette Score, Calinski-Harabasz Index, and Davies-Bouldin Index.
* **External Metrics (if labels exist):** Adjusted Rand Index (ARI), Normalized Mutual Information (NMI), and Fowlkes-Mallows Index.

### 3. Stability Analysis
* **Method:** Perform bootstrapping by subsampling 80% of the data.
* **Evaluation:** Compute the average similarity between different clusterings using the Adjusted Rand Index (ARI) to report stability scores.

### 4. Cluster Ensemble Method
* **Implementation:** Combine results from K-Means (various $k$), GMM, and DBSCAN using majority voting or a co-association matrix.
* **Comparison:** Evaluate the ensemble’s performance against individual algorithms using ARI/NMI.

## Visual Analysis & Discussion
* **Projections:** Visualize clusters using PCA or UMAP to highlight differences in cluster shapes (spherical vs. arbitrary).
* **Theoretical Discussion:** Analyze the underlying assumptions of each algorithm (e.g., density-based vs. Gaussian mixtures) and how violations affect results.
