# =============================================================================
# CENG 463 – Machine Learning – Take-Home Midterm
# Question 4
# Dataset: Wholesale Customer (primary) + Optdigits (external validation)
# =============================================================================

# ── Imports ──────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import warnings
import time
from itertools import combinations

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.mixture import GaussianMixture
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.neighbors import NearestNeighbors
from sklearn.metrics import (
    silhouette_score, calinski_harabasz_score, davies_bouldin_score,
    adjusted_rand_score, normalized_mutual_info_score, fowlkes_mallows_score,
    pairwise_distances
)
from sklearn.utils import resample

from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from scipy.stats import mode

try:
    import umap.umap_ as umap_module
    UMAP_AVAILABLE = True
except ImportError:
    try:
        import umap
        umap_module = umap
        UMAP_AVAILABLE = True
    except ImportError:
        UMAP_AVAILABLE = False
        print("[WARNING] UMAP not installed. Skipping UMAP visualisation.")

warnings.filterwarnings("ignore")
np.random.seed(42)

FIGSIZE_WIDE  = (16, 5)
FIGSIZE_SQ    = (10, 8)
FIGSIZE_TALL  = (12, 10)
PALETTE       = "tab10"


# =============================================================================
# 1. DATA LOADING
# =============================================================================

def load_wholesale():
    file_path = "Wholesale customers data.csv"

    df = pd.read_csv(file_path)

    # Fix the UCI dataset typo so the rest of the code works
    df.rename(columns={'Delicassen': 'Delicatessen'}, inplace=True)

    print(f"[Wholesale] Loaded from local file  → shape: {df.shape}")

    return df


def load_optdigits():
    """Load sklearn Digits dataset (8×8 pixel images of hand-written digits)."""
    digits = load_digits()
    X = digits.data
    y = digits.target
    print(f"[Optdigits] shape: {X.shape}  | classes: {np.unique(y)}")
    return X, y


# =============================================================================
# 2. EXPLORATORY DATA ANALYSIS
# =============================================================================

def eda_wholesale(df, save_prefix="figs/q4"):
    import os; os.makedirs("figs", exist_ok=True)

    print("\n" + "="*60)
    print("2. EXPLORATORY DATA ANALYSIS – Wholesale")
    print("="*60)
    print(df.describe().T.round(2))
    print(f"\nMissing values:\n{df.isnull().sum()}")

    feature_cols = ["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicatessen"]

    # Distribution plots
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    for ax, col in zip(axes.flat, feature_cols):
        ax.hist(df[col], bins=30, color="steelblue", edgecolor="white", alpha=0.85)
        ax.set_title(col); ax.set_xlabel("Annual spending (m.u.)")
    fig.suptitle("Feature Distributions – Wholesale Customer Dataset", fontsize=13, y=1.02)
    fig.tight_layout()
    plt.savefig(f"{save_prefix}_eda_distributions.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    corr = df[feature_cols].corr()
    sns.heatmap(corr, annot=True, fmt=".2f", cmap="coolwarm", ax=ax,
                linewidths=0.5, vmin=-1, vmax=1)
    ax.set_title("Correlation Heatmap – Wholesale Features")
    fig.tight_layout()
    plt.savefig(f"{save_prefix}_eda_correlation.png", dpi=120, bbox_inches="tight")
    plt.close()

    print("[EDA] Figures saved.")


    # =============================================================================
# 3. PREPROCESSING
# =============================================================================

def preprocess_wholesale(df):
    """Log-transform (skewed), drop Channel/Region, standardise."""
    feature_cols = ["Fresh","Milk","Grocery","Frozen","Detergents_Paper","Delicatessen"]
    X = df[feature_cols].copy()

    # Log-transform (add 1 to avoid log(0))
    X_log = np.log1p(X)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_log)
    print(f"[Preprocessing] Log-transform + StandardScaler applied  → shape: {X_scaled.shape}")
    return X_scaled, feature_cols


def preprocess_optdigits(X):
    scaler = StandardScaler()
    return scaler.fit_transform(X)



# =============================================================================
# 4. GAP STATISTIC (helper)
# =============================================================================

def gap_statistic(X, k_range, B=10, random_state=42):
    """
    Compute Gap Statistic for K-Means.
    Returns gaps, sk (std), and optimal k.
    """
    rng   = np.random.default_rng(random_state)
    gaps  = []
    sks   = []

    # Bounding box of X
    mins  = X.min(axis=0)
    maxs  = X.max(axis=0)

    for k in k_range:
        # Inertia on actual data
        km_real = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        km_real.fit(X)
        W_k = np.log(km_real.inertia_)

        # Inertia on B random reference datasets
        W_ref_list = []
        for _ in range(B):
            X_ref = rng.uniform(mins, maxs, size=X.shape)
            km_ref = KMeans(n_clusters=k, n_init=5, random_state=random_state)
            km_ref.fit(X_ref)
            W_ref_list.append(np.log(km_ref.inertia_))

        W_ref  = np.mean(W_ref_list)
        gap    = W_ref - W_k
        sd     = np.std(W_ref_list) * np.sqrt(1 + 1/B)

        gaps.append(gap)
        sks.append(sd)

    # Optimal k: smallest k such that Gap(k) >= Gap(k+1) - s(k+1)
    gaps = np.array(gaps)
    sks  = np.array(sks)
    opt_k = k_range[0]
    for i in range(len(k_range) - 1):
        if gaps[i] >= gaps[i+1] - sks[i+1]:
            opt_k = k_range[i]
            break
    else:
        opt_k = k_range[np.argmax(gaps)]

    return gaps, sks, opt_k



# =============================================================================
# 5. K-MEANS
# =============================================================================

def run_kmeans(X, k_range=range(2, 11), save_prefix="figs/q4"):
    print("\n" + "="*60)
    print("5. K-MEANS CLUSTERING")
    print("="*60)

    inertias     = []
    sil_scores   = []
    ch_scores    = []
    db_scores    = []

    for k in k_range:
        km = KMeans(n_clusters=k, n_init=15, random_state=42)
        labels = km.fit_predict(X)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X, labels))
        ch_scores.append(calinski_harabasz_score(X, labels))
        db_scores.append(davies_bouldin_score(X, labels))

    # Gap statistic
    print("  Computing Gap Statistic (B=20 reference datasets) …")
    gaps, sks, opt_k_gap = gap_statistic(X, list(k_range), B=20)

    # ── Plots ──────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    ks = list(k_range)

    axes[0].plot(ks, inertias, "bo-"); axes[0].set_title("Elbow – Inertia")
    axes[0].set_xlabel("k"); axes[0].set_ylabel("Inertia")

    axes[1].plot(ks, sil_scores, "go-"); axes[1].set_title("Silhouette Score")
    axes[1].set_xlabel("k")

    axes[2].errorbar(ks, gaps, yerr=sks, fmt="ro-", capsize=4)
    axes[2].set_title(f"Gap Statistic  (opt k={opt_k_gap})")
    axes[2].axvline(opt_k_gap, linestyle="--", color="grey")
    axes[2].set_xlabel("k")

    axes[3].plot(ks, ch_scores, "mo-"); axes[3].set_title("Calinski-Harabasz Index")
    axes[3].set_xlabel("k")

    for ax in axes: ax.grid(True, alpha=0.3)
    fig.suptitle("K-Means – Model Selection Criteria", fontsize=13)
    fig.tight_layout()
    plt.savefig(f"{save_prefix}_kmeans_selection.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Choose best k by silhouette
    best_k_sil = ks[np.argmax(sil_scores)]
    print(f"  Best k (silhouette) : {best_k_sil}")
    print(f"  Best k (gap stat.)  : {opt_k_gap}")

    km_final = KMeans(n_clusters=best_k_sil, n_init=20, random_state=42)
    labels_km = km_final.fit_predict(X)
    print(f"  KMeans (k={best_k_sil}) → Sil={silhouette_score(X,labels_km):.4f} | "
          f"CH={calinski_harabasz_score(X,labels_km):.2f} | "
          f"DB={davies_bouldin_score(X,labels_km):.4f}")

    return labels_km, best_k_sil, km_final



# =============================================================================
# 6. GAUSSIAN MIXTURE MODEL
# =============================================================================

def run_gmm(X, k_range=range(2, 11), save_prefix="figs/q4"):
    print("\n" + "="*60)
    print("6. GAUSSIAN MIXTURE MODEL")
    print("="*60)

    bics, aics = [], []

    for k in k_range:
        gmm = GaussianMixture(n_components=k, covariance_type="full",
                               n_init=5, random_state=42)
        gmm.fit(X)
        bics.append(gmm.bic(X))
        aics.append(gmm.aic(X))

    ks = list(k_range)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    axes[0].plot(ks, bics, "bs-"); axes[0].set_title("GMM – BIC"); axes[0].set_xlabel("Components")
    axes[1].plot(ks, aics, "rs-"); axes[1].set_title("GMM – AIC"); axes[1].set_xlabel("Components")
    for ax in axes: ax.grid(True, alpha=0.3)
    fig.suptitle("GMM – Component Selection via BIC / AIC", fontsize=13)
    fig.tight_layout()
    plt.savefig(f"{save_prefix}_gmm_selection.png", dpi=120, bbox_inches="tight")
    plt.close()

    best_k_bic = ks[np.argmin(bics)]
    best_k_aic = ks[np.argmin(aics)]
    print(f"  Best components (BIC): {best_k_bic} | (AIC): {best_k_aic}")

    gmm_final = GaussianMixture(n_components=best_k_bic, covariance_type="full",
                                 n_init=10, random_state=42)
    gmm_final.fit(X)
    labels_gmm = gmm_final.predict(X)

    print(f"  GMM (k={best_k_bic}) → Sil={silhouette_score(X,labels_gmm):.4f} | "
          f"CH={calinski_harabasz_score(X,labels_gmm):.2f} | "
          f"DB={davies_bouldin_score(X,labels_gmm):.4f}")

    return labels_gmm, best_k_bic, gmm_final



# =============================================================================
# 7. DBSCAN
# =============================================================================

def run_dbscan(X, save_prefix="figs/q4"):
    print("\n" + "="*60)
    print("7. DBSCAN")
    print("="*60)

    # k-distance graph for eps selection (k = 2*dim - 1 heuristic)
    k_dist = max(4, 2 * X.shape[1] - 1)
    nbrs   = NearestNeighbors(n_neighbors=k_dist).fit(X)
    distances, _ = nbrs.kneighbors(X)
    kth_dist = np.sort(distances[:, -1])[::-1]

    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(kth_dist, color="steelblue")
    ax.set_xlabel("Points (sorted)"); ax.set_ylabel(f"{k_dist}-NN Distance")
    ax.set_title(f"k-Distance Graph for DBSCAN eps Selection  (k={k_dist})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    plt.savefig(f"{save_prefix}_dbscan_kdistance.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Grid search over eps and min_samples
    # Use a wide range: from 10th to 95th percentile of k-distances
    eps_candidates   = np.percentile(kth_dist, [10, 20, 30, 50, 70, 85, 90, 95])
    min_samples_list = [3, 5, 7, 10]

    best_sil, best_eps, best_ms, best_labels = -1, None, None, None
    results = []
    for eps in eps_candidates:
        for ms in min_samples_list:
            db  = DBSCAN(eps=eps, min_samples=ms)
            lbl = db.fit_predict(X)
            n_clusters = len(set(lbl)) - (1 if -1 in lbl else 0)
            n_noise    = int(np.sum(lbl == -1))
            if n_clusters < 2:
                continue
            sil = silhouette_score(X, lbl)
            results.append({"eps": round(eps,4), "min_samples": ms,
                             "clusters": n_clusters, "noise": n_noise,
                             "silhouette": round(sil,4)})
            if sil > best_sil:
                best_sil, best_eps, best_ms, best_labels = sil, eps, ms, lbl

    if not results:
        print("  [DBSCAN] No valid configuration found; using eps=1.0, min_samples=3.")
        best_eps, best_ms = 1.0, 3
        best_labels = DBSCAN(eps=best_eps, min_samples=best_ms).fit_predict(X)

    if results:
        res_df = pd.DataFrame(results).sort_values("silhouette", ascending=False)
        print(res_df.head(10).to_string(index=False))
    print(f"\n  Best DBSCAN → eps={best_eps:.4f}, min_samples={best_ms}")
    n_cls = len(set(best_labels)) - (1 if -1 in best_labels else 0)
    print(f"  Clusters: {n_cls} | Noise pts: {np.sum(best_labels==-1)}")
    if n_cls >= 2:
        print(f"  Sil={silhouette_score(X, best_labels):.4f} | "
              f"CH={calinski_harabasz_score(X, best_labels):.2f} | "
              f"DB={davies_bouldin_score(X, best_labels):.4f}")

    return best_labels, best_eps, best_ms



# =============================================================================
# 8. AGGLOMERATIVE CLUSTERING
# =============================================================================

def run_agglomerative(X, n_clusters=3, save_prefix="figs/q4"):
    print("\n" + "="*60)
    print("8. AGGLOMERATIVE CLUSTERING (Ward Linkage)")
    print("="*60)

    # Dendrogram (use a subsample for readability)
    sample_idx = np.random.choice(len(X), size=min(200, len(X)), replace=False)
    X_sub = X[sample_idx]

    Z = linkage(X_sub, method="ward")
    fig, ax = plt.subplots(figsize=(14, 5))
    dendrogram(Z, ax=ax, truncate_mode="lastp", p=30,
               leaf_rotation=90, leaf_font_size=9, color_threshold=None)
    ax.set_title("Dendrogram – Ward Linkage (subsample of 200 pts)")
    ax.set_xlabel("Sample index"); ax.set_ylabel("Distance")
    fig.tight_layout()
    plt.savefig(f"{save_prefix}_agglomerative_dendrogram.png", dpi=120, bbox_inches="tight")
    plt.close()

    # Select n_clusters by evaluating silhouette
    sil_scores = []
    k_range    = range(2, 9)
    for k in k_range:
        agg = AgglomerativeClustering(n_clusters=k, linkage="ward")
        lbl = agg.fit_predict(X)
        sil_scores.append(silhouette_score(X, lbl))

    best_k = list(k_range)[np.argmax(sil_scores)]
    print(f"  Best k (silhouette): {best_k}")

    agg_final  = AgglomerativeClustering(n_clusters=best_k, linkage="ward")
    labels_agg = agg_final.fit_predict(X)

    print(f"  Agglomerative (k={best_k}) → "
          f"Sil={silhouette_score(X,labels_agg):.4f} | "
          f"CH={calinski_harabasz_score(X,labels_agg):.2f} | "
          f"DB={davies_bouldin_score(X,labels_agg):.4f}")

    return labels_agg, best_k



# =============================================================================
# 9. INTERNAL METRICS COMPARISON TABLE
# =============================================================================

def compare_internal_metrics(X, all_labels: dict):
    print("\n" + "="*60)
    print("9. INTERNAL METRICS COMPARISON")
    print("="*60)

    rows = []
    for name, lbl in all_labels.items():
        n_cls = len(set(lbl)) - (1 if -1 in lbl else 0)
        if n_cls < 2:
            rows.append({"Model": name, "Clusters": n_cls,
                         "Silhouette": np.nan, "CH-Index": np.nan, "DB-Index": np.nan})
            continue
        rows.append({
            "Model":     name,
            "Clusters":  n_cls,
            "Silhouette": round(silhouette_score(X, lbl), 4),
            "CH-Index":   round(calinski_harabasz_score(X, lbl), 2),
            "DB-Index":   round(davies_bouldin_score(X, lbl), 4),
        })

    df_metrics = pd.DataFrame(rows)
    print(df_metrics.to_string(index=False))
    return df_metrics



# =============================================================================
# 10. EXTERNAL METRICS (Optdigits)
# =============================================================================

def run_external_validation(X_opt, y_opt, save_prefix="figs/q4"):
    print("\n" + "="*60)
    print("10. EXTERNAL VALIDATION – Optdigits (10 true classes)")
    print("="*60)

    n_true_classes = len(np.unique(y_opt))

    algorithms = {
        "KMeans":             KMeans(n_clusters=n_true_classes, n_init=20, random_state=42),
        "GMM":                GaussianMixture(n_components=n_true_classes, n_init=5, random_state=42),
        "Agglomerative":      AgglomerativeClustering(n_clusters=n_true_classes, linkage="ward"),
        "DBSCAN":             DBSCAN(eps=4.5, min_samples=5),
    }

    rows = []
    for name, model in algorithms.items():
        if hasattr(model, "fit_predict"):
            lbl = model.fit_predict(X_opt)
        else:
            model.fit(X_opt)
            lbl = model.predict(X_opt)

        n_cls = len(set(lbl)) - (1 if -1 in lbl else 0)
        ari   = adjusted_rand_score(y_opt, lbl)
        nmi   = normalized_mutual_info_score(y_opt, lbl)
        fmi   = fowlkes_mallows_score(y_opt, lbl)

        rows.append({"Model": name, "Clusters": n_cls,
                     "ARI": round(ari,4), "NMI": round(nmi,4), "FMI": round(fmi,4)})
        print(f"  {name:<18} | clusters={n_cls:2d} | ARI={ari:.4f} | "
              f"NMI={nmi:.4f} | FMI={fmi:.4f}")

    df_ext = pd.DataFrame(rows)
    return df_ext



# =============================================================================
# 11. CLUSTER STABILITY ANALYSIS (Bootstrap)
# =============================================================================

def stability_analysis(X, n_clusters_dict: dict, n_bootstrap=20, subsample_frac=0.8,
                        save_prefix="figs/q4"):
    """
    For each algorithm, subsample 80% of data n_bootstrap times,
    cluster, and compute average ARI between pairs of runs.
    """
    print("\n" + "="*60)
    print("11. CLUSTER STABILITY ANALYSIS")
    print("="*60)

    results = {}

    def _cluster(X_sub, name, k):
        if name == "KMeans":
            return KMeans(n_clusters=k, n_init=10, random_state=None).fit_predict(X_sub)
        elif name == "GMM":
            m = GaussianMixture(n_components=k, n_init=3)
            m.fit(X_sub)
            return m.predict(X_sub)
        elif name == "Agglomerative":
            return AgglomerativeClustering(n_clusters=k, linkage="ward").fit_predict(X_sub)
        elif name == "DBSCAN":
            return DBSCAN(eps=k[0], min_samples=k[1]).fit_predict(X_sub)

    for name, param in n_clusters_dict.items():
        aris     = []
        all_lbl  = []
        indices  = []
        n        = len(X)

        for _ in range(n_bootstrap):
            idx = resample(np.arange(n), replace=False,
                           n_samples=int(n * subsample_frac), random_state=None)
            idx = np.sort(idx)
            lbl = _cluster(X[idx], name, param)
            all_lbl.append(lbl)
            indices.append(idx)

        # Pairwise ARI on common samples
        pair_aris = []
        for i, j in combinations(range(n_bootstrap), 2):
            common = np.intersect1d(indices[i], indices[j])
            if len(common) < 10:
                continue
            pos_i = np.searchsorted(indices[i], common)
            pos_j = np.searchsorted(indices[j], common)
            ari   = adjusted_rand_score(all_lbl[i][pos_i], all_lbl[j][pos_j])
            pair_aris.append(ari)

        mean_ari = np.mean(pair_aris) if pair_aris else np.nan
        std_ari  = np.std(pair_aris)  if pair_aris else np.nan
        results[name] = {"mean_ARI": round(mean_ari, 4), "std_ARI": round(std_ari, 4)}
        print(f"  {name:<18} → Stability ARI = {mean_ari:.4f} ± {std_ari:.4f}")

    # Bar plot
    names  = list(results.keys())
    means  = [results[n]["mean_ARI"] for n in names]
    stds   = [results[n]["std_ARI"]  for n in names]

    fig, ax = plt.subplots(figsize=(8, 4))
    bars = ax.bar(names, means, yerr=stds, color="steelblue", edgecolor="white",
                  capsize=5, alpha=0.85)
    ax.set_ylabel("Mean ARI (higher = more stable)")
    ax.set_title("Cluster Stability Analysis (80% Bootstrap, 20 runs)")
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", alpha=0.3)
    for bar, m in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, m + 0.02, f"{m:.3f}",
                ha="center", va="bottom", fontsize=10)
    fig.tight_layout()
    plt.savefig(f"{save_prefix}_stability.png", dpi=120, bbox_inches="tight")
    plt.close()

    return results



# =============================================================================
# 12. CLUSTER ENSEMBLE (Co-Association Matrix)
# =============================================================================

def cluster_ensemble(X, n_members=10, k_range=(2, 6),
                     gmm_k=3, dbscan_params=(0.8, 5),
                     y_true=None, save_prefix="figs/q4"):
    """
    Build a co-association matrix from K-Means (varying k), GMM, and DBSCAN.
    Final clustering via Agglomerative on the dissimilarity matrix.
    """
    print("\n" + "="*60)
    print("12. CLUSTER ENSEMBLE (Co-Association Matrix)")
    print("="*60)

    n = len(X)
    coassoc = np.zeros((n, n))
    n_partitions = 0

    # --- K-Means with different k ---
    for k in range(k_range[0], k_range[1]+1):
        for seed in range(3):           # 3 runs per k for robustness
            lbl = KMeans(n_clusters=k, n_init=5, random_state=seed).fit_predict(X)
            for c in np.unique(lbl):
                mask = (lbl == c)
                idx  = np.where(mask)[0]
                coassoc[np.ix_(idx, idx)] += 1
            n_partitions += 1

    # --- GMM ---
    for seed in range(3):
        m   = GaussianMixture(n_components=gmm_k, n_init=3, random_state=seed)
        lbl = m.fit_predict(X)
        for c in np.unique(lbl):
            idx = np.where(lbl == c)[0]
            coassoc[np.ix_(idx, idx)] += 1
        n_partitions += 1

    # --- DBSCAN (treat noise as singleton clusters) ---
    lbl_db = DBSCAN(eps=dbscan_params[0], min_samples=dbscan_params[1]).fit_predict(X)
    for c in np.unique(lbl_db):
        idx = np.where(lbl_db == c)[0]
        coassoc[np.ix_(idx, idx)] += 1
    n_partitions += 1

    # Normalise → co-occurrence probability
    coassoc /= n_partitions

    # Dissimilarity matrix
    dissim = 1.0 - coassoc
    np.fill_diagonal(dissim, 0)

    # Condensed form for agglomerative
    condensed = squareform(dissim, checks=False)

    # Choose n_ensemble_clusters by silhouette on condensed distances
    best_sil, best_k_ens, best_lbl_ens = -1, 2, None
    for k in range(2, 8):
        agg = AgglomerativeClustering(n_clusters=k, metric="precomputed", linkage="average")
        lbl = agg.fit_predict(dissim)
        if len(np.unique(lbl)) < 2:
            continue
        # Silhouette on the original feature space
        sil = silhouette_score(X, lbl)
        if sil > best_sil:
            best_sil, best_k_ens, best_lbl_ens = sil, k, lbl

    print(f"  Ensemble: {n_partitions} base partitions combined")
    print(f"  Best k = {best_k_ens}  (Sil = {best_sil:.4f})")
    print(f"  Ensemble Sil={silhouette_score(X,best_lbl_ens):.4f} | "
          f"CH={calinski_harabasz_score(X,best_lbl_ens):.2f} | "
          f"DB={davies_bouldin_score(X,best_lbl_ens):.4f}")

    if y_true is not None:
        ari = adjusted_rand_score(y_true, best_lbl_ens)
        nmi = normalized_mutual_info_score(y_true, best_lbl_ens)
        print(f"  ARI={ari:.4f}  NMI={nmi:.4f}")

    # Heatmap of co-association matrix (subsample 100 pts)
    sample = np.random.choice(n, size=min(150, n), replace=False)
    sample_sorted = sample[np.argsort(best_lbl_ens[sample])]

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(coassoc[np.ix_(sample_sorted, sample_sorted)],
                   cmap="hot", aspect="auto", vmin=0, vmax=1)
    plt.colorbar(im, ax=ax, label="Co-occurrence probability")
    ax.set_title("Co-Association Matrix (150-pt subsample, sorted by cluster)")
    ax.set_xlabel("Sample"); ax.set_ylabel("Sample")
    fig.tight_layout()
    plt.savefig(f"{save_prefix}_ensemble_coassoc.png", dpi=120, bbox_inches="tight")
    plt.close()

    return best_lbl_ens, best_k_ens



# =============================================================================
# 13. VISUALISATION – PCA & UMAP
# =============================================================================

def visualise_clusters(X, labels_dict: dict, save_prefix="figs/q4"):
    print("\n" + "="*60)
    print("13. VISUALISATION – PCA & UMAP Projections")
    print("="*60)

    # PCA 2D
    pca     = PCA(n_components=2, random_state=42)
    X_pca   = pca.fit_transform(X)
    var_exp = pca.explained_variance_ratio_

    n_methods  = len(labels_dict)
    fig, axes  = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
    if n_methods == 1:
        axes = [axes]

    for ax, (name, lbl) in zip(axes, labels_dict.items()):
        unique_lbl = np.unique(lbl)
        cmap       = plt.cm.get_cmap(PALETTE, len(unique_lbl))
        for i, c in enumerate(unique_lbl):
            mask = (lbl == c)
            label_str = f"Noise" if c == -1 else f"C{c}"
            ax.scatter(X_pca[mask, 0], X_pca[mask, 1],
                       s=12, alpha=0.6, color=cmap(i), label=label_str)
        ax.set_title(f"{name}")
        ax.set_xlabel(f"PC1 ({var_exp[0]*100:.1f}%)")
        ax.set_ylabel(f"PC2 ({var_exp[1]*100:.1f}%)")
        ax.legend(fontsize=7, markerscale=2, loc="best")
        ax.grid(True, alpha=0.2)

    fig.suptitle("PCA 2D – Cluster Visualisation", fontsize=13)
    fig.tight_layout()
    plt.savefig(f"{save_prefix}_pca_clusters.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  PCA projection saved.")

    # UMAP 2D
    if UMAP_AVAILABLE:
        try:
            reducer   = umap_module.UMAP(n_components=2, n_neighbors=15,
                                          min_dist=0.1, random_state=42)
            X_umap    = reducer.fit_transform(X)

            fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))
            if n_methods == 1:
                axes = [axes]

            for ax, (name, lbl) in zip(axes, labels_dict.items()):
                unique_lbl = np.unique(lbl)
                cmap       = plt.cm.get_cmap(PALETTE, len(unique_lbl))
                for i, c in enumerate(unique_lbl):
                    mask = (lbl == c)
                    label_str = f"Noise" if c == -1 else f"C{c}"
                    ax.scatter(X_umap[mask, 0], X_umap[mask, 1],
                               s=12, alpha=0.6, color=cmap(i), label=label_str)
                ax.set_title(f"{name}")
                ax.set_xlabel("UMAP-1"); ax.set_ylabel("UMAP-2")
                ax.legend(fontsize=7, markerscale=2, loc="best")
                ax.grid(True, alpha=0.2)

            fig.suptitle("UMAP 2D – Cluster Visualisation", fontsize=13)
            fig.tight_layout()
            plt.savefig(f"{save_prefix}_umap_clusters.png", dpi=120, bbox_inches="tight")
            plt.close()
            print("  UMAP projection saved.")
        except Exception as e:
            print(f"  [UMAP error] {e}")
    else:
        print("  UMAP not available; skipping UMAP projection.")


# =============================================================================
# 14. SUMMARY TABLE
# =============================================================================

def print_summary_table(internal_df, stability_results,
                        internal_filepath="internal_metrics.csv",
                        stability_filepath="stability_results.csv"):

    print("\n" + "="*60)
    print("14. FINAL SUMMARY")
    print("="*60)
    print("\nInternal Metrics:")
    print(internal_df.to_string(index=False))

    print("\nStability (ARI):")
    for name, vals in stability_results.items():
        print(f"  {name:<18}: {vals['mean_ARI']:.4f} ± {vals['std_ARI']:.4f}")

    # Save the internal metrics DataFrame to its own file
    internal_df.to_csv(internal_filepath, index=False)

    # Convert the stability dictionary into a pandas DataFrame
    stability_data = []
    for name, vals in stability_results.items():
        stability_data.append({
            "Method": name,
            "Mean_ARI": vals["mean_ARI"],
            "Std_ARI": vals["std_ARI"]
        })
    stability_df = pd.DataFrame(stability_data)

    # Save the stability DataFrame to a completely separate file
    stability_df.to_csv(stability_filepath, index=False)



# =============================================================================
# MAIN PIPELINE
# =============================================================================

def main():
    import os; os.makedirs("figs", exist_ok=True)

    # ── Load & preprocess Wholesale ─────────────────────────────────────────
    df_raw  = load_wholesale()
    X, feat = preprocess_wholesale(df_raw)

    # ── EDA ─────────────────────────────────────────────────────────────────
    eda_wholesale(df_raw)

    # ── Clustering algorithms on Wholesale ──────────────────────────────────
    labels_km,  best_k_km,  km_model  = run_kmeans(X)
    labels_gmm, best_k_gmm, gmm_model = run_gmm(X)
    labels_db,  best_eps,   best_ms   = run_dbscan(X)
    labels_agg, best_k_agg            = run_agglomerative(X, n_clusters=best_k_km)

    all_labels = {
        "KMeans":        labels_km,
        "GMM":           labels_gmm,
        "DBSCAN":        labels_db,
        "Agglomerative": labels_agg,
    }

    # ── Internal metrics ─────────────────────────────────────────────────────
    internal_df = compare_internal_metrics(X, all_labels)

    # ── Stability analysis ───────────────────────────────────────────────────
    # DBSCAN param passed as tuple (eps, min_samples)
    stability_params = {
        "KMeans":        best_k_km,
        "GMM":           best_k_gmm,
        "Agglomerative": best_k_agg,
        "DBSCAN":        (best_eps, best_ms),
    }
    stability_results = stability_analysis(X, stability_params, n_bootstrap=20)

    # ── Cluster ensemble ─────────────────────────────────────────────────────
    labels_ens, k_ens = cluster_ensemble(
        X,
        k_range=(2, max(best_k_km, best_k_gmm, 4)),
        gmm_k=best_k_gmm,
        dbscan_params=(best_eps, best_ms),
        y_true=None,
    )
    all_labels["Ensemble"] = labels_ens

    # Recompute metrics with ensemble included
    internal_df = compare_internal_metrics(X, all_labels)

    # ── Visualisation ────────────────────────────────────────────────────────
    visualise_clusters(X, all_labels)

    # ── External validation (Optdigits) ──────────────────────────────────────
    X_opt, y_opt = load_optdigits()
    X_opt_scaled = preprocess_optdigits(X_opt)
    ext_df = run_external_validation(X_opt_scaled, y_opt)

    ext_df.to_csv("external_validation_metrics.csv", index=False)

    df_final_clusters = df_raw.copy()
    for model_name, cluster_labels in all_labels.items():
        df_final_clusters[f"{model_name}_Cluster"] = cluster_labels
    df_final_clusters.to_csv("wholesale_cluster_assignments.csv", index=False)

    # PCA/UMAP for Optdigits (best algorithm: KMeans)
    n_cls_opt = len(np.unique(y_opt))
    lbl_opt_km = KMeans(n_clusters=n_cls_opt, n_init=20, random_state=42).fit_predict(X_opt_scaled)
    pca2 = PCA(n_components=2, random_state=42)
    X_opt_pca = pca2.fit_transform(X_opt_scaled)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for label_arr, title, ax in [
        (y_opt,     "True Labels",        axes[0]),
        (lbl_opt_km,"KMeans Prediction",  axes[1]),
    ]:
        cmap = plt.cm.get_cmap(PALETTE, 10)
        for c in np.unique(label_arr):
            mask = (label_arr == c)
            ax.scatter(X_opt_pca[mask,0], X_opt_pca[mask,1],
                       s=8, alpha=0.6, color=cmap(c), label=str(c))
        ax.set_title(f"Optdigits PCA – {title}")
        ax.set_xlabel("PC1"); ax.set_ylabel("PC2")
        ax.legend(fontsize=6, markerscale=2, loc="best", ncol=2)
        ax.grid(True, alpha=0.2)
    fig.tight_layout()
    plt.savefig("figs/q4_optdigits_pca.png", dpi=120, bbox_inches="tight")
    plt.close()
    print("  Optdigits PCA visualisation saved.")


    # ── Final summary ─────────────────────────────────────────────────────
    print_summary_table(internal_df, stability_results)



    print("\n" + "="*60)
    print("All figures saved to  ./figs/")
    print("="*60)


if __name__ == "__main__":
    main()
