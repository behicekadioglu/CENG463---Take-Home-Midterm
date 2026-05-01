# Question 3: Dimensionality Reduction – Autoencoders and Manifold Learning

## 🎯 Objective
The objective of this task is to compare linear, non-linear, and deep learning-based dimensionality reduction techniques, evaluating their performance through reconstruction quality and downstream classification tasks.

## 📊 Dataset
* **Dataset Used:** MNIST.

## 🛠️ Implementation Tasks

### 1. Dimensionality Reduction Methods
Implement and compare five distinct methods:
* **PCA:** Standard linear reduction.
* **Kernel PCA:** Non-linear reduction using an RBF kernel.
* **t-SNE:** Manifold learning with a perplexity grid search (values: 5, 30, 50).
* **UMAP:** Manifold learning with tuned `n_neighbors` and `min_dist` parameters.
* **Autoencoder:** A simple undercomplete autoencoder with 3 hidden layers and a bottleneck dimension of 2 or 3, trained using reconstruction loss.

### 2. Quantitative Evaluation
* **Reconstruction Error:** MSE calculated for PCA, Kernel PCA, and the autoencoder on the test set.
* **Quality Scores:** Trustworthiness and continuity scores for t-SNE and UMAP.
* **Downstream Task:** Training a k-NN classifier (k=5) on the reduced 2D/3D space and reporting 5-fold cross-validation accuracy.
* **Stress Metric:** Computation of Kruskal's stress for t-SNE and UMAP.

### 3. Visual Analysis
* **Embeddings:** 2D visualizations of embeddings for all methods, colored by class to compare class separation.
* **Latent Space:** Visual analysis of the autoencoder's latent space to determine if it captures semantic structures such as digit style or thickness.

## 🔍 Analysis & Discussion
* **Computational Complexity:** Discussion of the time and memory requirements for each method.
* **Comparative Performance:** Analysis of scenarios where autoencoders might outperform traditional methods like PCA or t-SNE.
