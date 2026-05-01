"""
CENG 463 – Machine Learning Take-Home Midterm
Question 3

Dataset: MNIST (auto-downloaded via torchvision / sklearn)
"""

# ─────────────────────────────────────────────────────────────────────────────
# 0.  Imports & Reproducibility
# ─────────────────────────────────────────────────────────────────────────────
import warnings
warnings.filterwarnings("ignore")

import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import fetch_openml
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import mean_squared_error
from sklearn.manifold import trustworthiness          # sklearn >= 1.0
from sklearn.decomposition import KernelPCA

import umap

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

SEED = 22
np.random.seed(SEED)
torch.manual_seed(SEED)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")


# ─────────────────────────────────────────────────────────────────────────────
# 1.  Load MNIST
# ─────────────────────────────────────────────────────────────────────────────

mnist = fetch_openml("mnist_784", version=1, as_frame=False, parser="auto")
X_full, y_full = mnist.data.astype(np.float32), mnist.target.astype(int)

# Subsample for t-SNE / UMAP (they are slow on 70 k samples)
N_SAMPLE = 10_000
idx = np.random.RandomState(SEED).choice(len(X_full), N_SAMPLE, replace=False)
X, y = X_full[idx], y_full[idx]

# Normalise to [0, 1]
X = X / 255.0

print(f"Full dataset   : {X_full.shape}")
print(f"Working subset : {X.shape}  (used for manifold methods)")
print(f"Classes        : {np.unique(y)}")


# ─────────────────────────────────────────────────────────────────────────────
# 2.  Helpers
# ─────────────────────────────────────────────────────────────────────────────
def knn_accuracy(X_embed, y, n_splits=5, k=5):
    """5-fold stratified CV accuracy with k-NN on the reduced space."""
    knn = KNeighborsClassifier(n_neighbors=k)
    cv  = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    scores = cross_val_score(knn, X_embed, y, cv=cv, scoring="accuracy")
    return scores.mean(), scores.std()

def reconstruction_mse(X_orig, X_recon):
    return mean_squared_error(X_orig, X_recon)

def kruskal_stress(X_orig, X_embed):
    """Kruskal's Stress-1 on a subsample for speed."""
    n = min(2000, len(X_orig))
    idx_s = np.random.choice(len(X_orig), n, replace=False)
    X_o, X_e = X_orig[idx_s], X_embed[idx_s]
    d_orig  = np.sqrt(((X_o[:, None] - X_o[None]) ** 2).sum(-1))
    d_embed = np.sqrt(((X_e[:, None] - X_e[None]) ** 2).sum(-1))
    num  = ((d_orig - d_embed) ** 2).sum()
    den  = (d_orig ** 2).sum()
    return np.sqrt(num / (den + 1e-12))

def plot_2d_embedding(X_2d, y, title, ax, alpha=0.4, s=3):
    cmap = plt.cm.get_cmap("tab10", 10)
    for cls in range(10):
        mask = y == cls
        ax.scatter(X_2d[mask, 0], X_2d[mask, 1],
                   color=cmap(cls), label=str(cls),
                   alpha=alpha, s=s)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])

# Storage for summary
results = []


# ─────────────────────────────────────────────────────────────────────────────
# 3.  PCA
# ─────────────────────────────────────────────────────────────────────────────

t0 = time.time()
pca = PCA(n_components=2, random_state=SEED)
X_pca = pca.fit_transform(X)
t_pca = time.time() - t0

# Reconstruction via inverse_transform
X_pca_recon = pca.inverse_transform(X_pca)
mse_pca = reconstruction_mse(X, X_pca_recon)

acc_pca, std_pca = knn_accuracy(X_pca, y)
tw_pca = trustworthiness(X, X_pca, n_neighbors=5)

print(f"  Time            : {t_pca:.2f}s")
print(f"  Recon MSE       : {mse_pca:.4f}")
print(f"  Trustworthiness : {tw_pca:.4f}")
print(f"  k-NN Accuracy   : {acc_pca:.4f} ± {std_pca:.4f}")
print(f"  Explained Var.  : {pca.explained_variance_ratio_.sum()*100:.1f}%")

results.append({
    "Method": "PCA", "Time (s)": round(t_pca, 2),
    "Recon MSE": round(mse_pca, 4), "Trustworthiness": round(tw_pca, 4),
    "Continuity": "N/A", "Kruskal Stress": "N/A",
    "k-NN Acc.": f"{acc_pca:.4f}±{std_pca:.4f}"
})


# ─────────────────────────────────────────────────────────────────────────────
# 4.  Kernel PCA  (RBF)
# ─────────────────────────────────────────────────────────────────────────────

t0 = time.time()
kpca = KernelPCA(n_components=2, kernel="rbf", gamma=0.01,
                 fit_inverse_transform=True, random_state=SEED)
X_kpca = kpca.fit_transform(X)
t_kpca = time.time() - t0

X_kpca_recon = kpca.inverse_transform(X_kpca)
mse_kpca = reconstruction_mse(X, X_kpca_recon)
acc_kpca, std_kpca = knn_accuracy(X_kpca, y)
tw_kpca = trustworthiness(X, X_kpca, n_neighbors=5)

print(f"  Time            : {t_kpca:.2f}s")
print(f"  Recon MSE       : {mse_kpca:.4f}")
print(f"  Trustworthiness : {tw_kpca:.4f}")
print(f"  k-NN Accuracy   : {acc_kpca:.4f} ± {std_kpca:.4f}")

results.append({
    "Method": "Kernel PCA (RBF)", "Time (s)": round(t_kpca, 2),
    "Recon MSE": round(mse_kpca, 4), "Trustworthiness": round(tw_kpca, 4),
    "Continuity": "N/A", "Kruskal Stress": "N/A",
    "k-NN Acc.": f"{acc_kpca:.4f}±{std_kpca:.4f}"
})


# ─────────────────────────────────────────────────────────────────────────────
# 5.  t-SNE  (perplexity grid: 5, 30, 50)
# ─────────────────────────────────────────────────────────────────────────────

tsne_embeddings = {}
tsne_scores = {}


for perp in [5, 30, 50]:
    t0 = time.time()
    tsne = TSNE(n_components=2, perplexity=perp, n_iter=1000,
                random_state=SEED, n_jobs=-1)
    X_tsne = tsne.fit_transform(X)
    t_tsne = time.time() - t0

    tw_tsne  = trustworthiness(X, X_tsne, n_neighbors=5)
    stress   = kruskal_stress(X, X_tsne)
    acc_tsne, std_tsne = knn_accuracy(X_tsne, y)

    print(f"  perplexity={perp:<3}  Time={t_tsne:.1f}s  "
          f"TW={tw_tsne:.4f}  Stress={stress:.4f}  "
          f"k-NN={acc_tsne:.4f}±{std_tsne:.4f}")

    tsne_embeddings[perp] = X_tsne
    tsne_scores[perp] = acc_tsne

    results.append({
        "Method": f"t-SNE (perp={perp})", "Time (s)": round(t_tsne, 2),
        "Recon MSE": "N/A", "Trustworthiness": round(tw_tsne, 4),
        "Continuity": "N/A", "Kruskal Stress": round(stress, 4),
        "k-NN Acc.": f"{acc_tsne:.4f}±{std_tsne:.4f}"
    })

best_perp = max(tsne_scores, key=tsne_scores.get)

print(f"\nAutomatically selected best perplexity based on k-NN Accuracy: {best_perp}")

X_tsne_best = tsne_embeddings[best_perp]


# ─────────────────────────────────────────────────────────────────────────────
# 6.  UMAP  (tune n_neighbors and min_dist)
# ─────────────────────────────────────────────────────────────────────────────

umap_grid = [(15, 0.1), (15, 0.5), (30, 0.1), (30, 0.5)]
umap_embeddings = {}
umap_scores = {}

for nn, md in umap_grid:
    t0 = time.time()
    reducer = umap.UMAP(n_components=2, n_neighbors=nn, min_dist=md,
                        random_state=SEED)
    X_umap = reducer.fit_transform(X)
    t_umap = time.time() - t0

    tw_umap  = trustworthiness(X, X_umap, n_neighbors=5)
    stress   = kruskal_stress(X, X_umap)
    acc_umap, std_umap = knn_accuracy(X_umap, y)

    print(f"  n_neighbors={nn}  min_dist={md}  Time={t_umap:.1f}s  "
          f"TW={tw_umap:.4f}  Stress={stress:.4f}  "
          f"k-NN={acc_umap:.4f}±{std_umap:.4f}")

    umap_embeddings[(nn, md)] = X_umap
    umap_scores[(nn, md)] = acc_umap

    results.append({
        "Method": f"UMAP (nn={nn}, md={md})", "Time (s)": round(t_umap, 2),
        "Recon MSE": "N/A", "Trustworthiness": round(tw_umap, 4),
        "Continuity": "N/A", "Kruskal Stress": round(stress, 4),
        "k-NN Acc.": f"{acc_umap:.4f}±{std_umap:.4f}"
    })

best_params = max(umap_scores, key=umap_scores.get)

print(f"\nAutomatically selected best UMAP configuration: n_neighbors={best_params[0]}, min_dist={best_params[1]}")

X_umap_best = umap_embeddings[best_params]


# ─────────────────────────────────────────────────────────────────────────────
# 7.  Undercomplete Autoencoder  (PyTorch)
# ─────────────────────────────────────────────────────────────────────────────

import torch
import torch.nn as nn
import torch.optim as optim

INPUT_DIM  = 784
LATENT_DIM = 2       # bottleneck
HIDDEN     = [512, 256, 64]
EPOCHS     = 50
BATCH_SIZE = 256
LR         = 1e-3

class Autoencoder(nn.Module):
    def __init__(self, input_dim=784, hidden=[512, 256, 64], latent=2):
        super().__init__()
        # Encoder
        enc_layers = []
        in_dim = input_dim
        for h in hidden:
            enc_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        enc_layers.append(nn.Linear(in_dim, latent))
        self.encoder = nn.Sequential(*enc_layers)

        # Decoder
        dec_layers = []
        in_dim = latent
        for h in reversed(hidden):
            dec_layers += [nn.Linear(in_dim, h), nn.BatchNorm1d(h), nn.ReLU()]
            in_dim = h
        dec_layers += [nn.Linear(in_dim, input_dim), nn.Sigmoid()]
        self.decoder = nn.Sequential(*dec_layers)

    def forward(self, x):
        z = self.encoder(x)
        return self.decoder(z), z

# Prepare data loaders
X_tensor = torch.tensor(X, dtype=torch.float32)
y_tensor = torch.tensor(y, dtype=torch.long)
dataset  = TensorDataset(X_tensor, y_tensor)

# 80/20 split for train/test
n_train = int(0.8 * len(dataset))
n_test  = len(dataset) - n_train
train_ds, test_ds = torch.utils.data.random_split(
    dataset, [n_train, n_test],
    generator=torch.Generator().manual_seed(SEED)
)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
test_loader  = DataLoader(test_ds,  batch_size=BATCH_SIZE, shuffle=False)

model     = Autoencoder(INPUT_DIM, HIDDEN, LATENT_DIM).to(DEVICE)
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=LR)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.5)

train_losses, test_losses = [], []

t0 = time.time()
for epoch in range(EPOCHS):
    # ── Train ──
    model.train()
    running = 0.0
    for xb, _ in train_loader:
        xb = xb.to(DEVICE)
        recon, _ = model(xb)
        loss = criterion(recon, xb)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        running += loss.item() * len(xb)
    train_losses.append(running / n_train)

    # ── Validation ──
    model.eval()
    with torch.no_grad():
        val_loss = sum(
            criterion(model(xb.to(DEVICE))[0], xb.to(DEVICE)).item() * len(xb)
            for xb, _ in test_loader
        ) / n_test
    test_losses.append(val_loss)
    scheduler.step()

    if (epoch + 1) % 10 == 0:
        print(f"  Epoch {epoch+1:3d}/{EPOCHS}  "
              f"train_loss={train_losses[-1]:.5f}  "
              f"val_loss={test_losses[-1]:.5f}")

t_ae = time.time() - t0

# ── Extract latent embeddings ──
model.eval()
with torch.no_grad():
    recon_out, latent_out = model(X_tensor.to(DEVICE))
    X_ae      = latent_out.cpu().numpy()
    X_ae_recon = recon_out.cpu().numpy()

mse_ae   = reconstruction_mse(X, X_ae_recon)
tw_ae    = trustworthiness(X, X_ae, n_neighbors=5)
acc_ae, std_ae = knn_accuracy(X_ae, y)

print(f"\n  Time            : {t_ae:.1f}s")
print(f"  Recon MSE       : {mse_ae:.4f}")
print(f"  Trustworthiness : {tw_ae:.4f}")
print(f"  k-NN Accuracy   : {acc_ae:.4f} ± {std_ae:.4f}")

results.append({
    "Method": "Autoencoder", "Time (s)": round(t_ae, 2),
    "Recon MSE": round(mse_ae, 4), "Trustworthiness": round(tw_ae, 4),
    "Continuity": "N/A", "Kruskal Stress": "N/A",
    "k-NN Acc.": f"{acc_ae:.4f}±{std_ae:.4f}"
})



# ─────────────────────────────────────────────────────────────────────────────
# 8.  Autoencoder Training Curve
# ─────────────────────────────────────────────────────────────────────────────

fig, ax = plt.subplots(figsize=(8, 4))
ax.plot(train_losses, label="Train Loss")
ax.plot(test_losses,  label="Val Loss")
ax.set_xlabel("Epoch"); ax.set_ylabel("MSE Loss")
ax.set_title("Autoencoder Training / Validation Loss")
ax.legend()
plt.tight_layout()
plt.savefig("ae_training_curve.png", dpi=150)
plt.close()
print("\nSaved: ae_training_curve.png")


# ─────────────────────────────────────────────────────────────────────────────
# 9.  2D Embedding Visualisations (all methods)
# ─────────────────────────────────────────────────────────────────────────────

embeddings_to_plot = {
    "PCA"              : X_pca,
    "Kernel PCA (RBF)" : X_kpca,
    "t-SNE (perp=5)"   : tsne_embeddings[5],
    "t-SNE (perp=30)"  : tsne_embeddings[30],
    "t-SNE (perp=50)"  : tsne_embeddings[50],
    "UMAP (nn=15,md=0.1)": umap_embeddings[(15, 0.1)],
    "UMAP (nn=30,md=0.5)": umap_embeddings[(30, 0.5)],
    "Autoencoder"      : X_ae,
}

n_plots = len(embeddings_to_plot)
ncols   = 4
nrows   = (n_plots + ncols - 1) // ncols

fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 4 * nrows))
axes = axes.flatten()

for i, (title, emb) in enumerate(embeddings_to_plot.items()):
    plot_2d_embedding(emb, y, title, axes[i])

# Shared legend
handles = [plt.Line2D([0], [0], marker="o", color="w",
           markerfacecolor=plt.cm.tab10(c / 10), markersize=7, label=str(c))
           for c in range(10)]
fig.legend(handles=handles, title="Digit", loc="lower right",
           ncol=5, fontsize=9, bbox_to_anchor=(1.0, 0.0))

for j in range(i + 1, len(axes)):
    axes[j].set_visible(False)

plt.suptitle("2D Embeddings – MNIST (coloured by digit class)", fontsize=13, y=1.01)
plt.tight_layout()
plt.savefig("all_embeddings_2d.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: all_embeddings_2d.png")


# ─────────────────────────────────────────────────────────────────────────────
# 10.  Autoencoder Latent Space – Semantic Analysis
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: latent scatter coloured by digit
cmap = plt.cm.get_cmap("tab10", 10)
for cls in range(10):
    mask = y == cls
    axes[0].scatter(X_ae[mask, 0], X_ae[mask, 1],
                    color=cmap(cls), label=str(cls),
                    alpha=0.5, s=5)
axes[0].set_title("Autoencoder Latent Space (z₁ vs z₂)")
axes[0].legend(title="Digit", markerscale=3, fontsize=8)
axes[0].set_xlabel("z₁"); axes[0].set_ylabel("z₂")

# Right: show reconstructed vs original for 5 digits
n_show = 10
sample_idx = [np.where(y == d)[0][0] for d in range(n_show)]
originals  = X[sample_idx]
reconst    = X_ae_recon[sample_idx]

grid = np.vstack([
    np.hstack([orig.reshape(28, 28) for orig in originals]),
    np.hstack([rec.reshape(28, 28)  for rec  in reconst]),
])
axes[1].imshow(grid, cmap="gray", vmin=0, vmax=1)
axes[1].set_title("Original (top) vs Reconstructed (bottom)")
axes[1].axis("off")

plt.suptitle("Autoencoder – Latent Space & Reconstruction Quality", fontsize=12)
plt.tight_layout()
plt.savefig("ae_latent_space.png", dpi=150)
plt.close()
print("Saved: ae_latent_space.png")


# ─────────────────────────────────────────────────────────────────────────────
# 11.  t-SNE Perplexity Grid – Side-by-side
# ─────────────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, perp in zip(axes, [5, 30, 50]):
    plot_2d_embedding(tsne_embeddings[perp], y,
                      f"t-SNE  perplexity={perp}", ax)
plt.suptitle("Effect of Perplexity on t-SNE Embedding", fontsize=12)
plt.tight_layout()
plt.savefig("tsne_perplexity_grid.png", dpi=150)
plt.close()
print("Saved: tsne_perplexity_grid.png")


# ─────────────────────────────────────────────────────────────────────────────
# 12.  Reconstruction Error Comparison  (PCA, Kernel PCA, Autoencoder)
# ─────────────────────────────────────────────────────────────────────────────

recon_methods = ["PCA", "Kernel PCA (RBF)", "Autoencoder"]
recon_mses    = [mse_pca, mse_kpca, mse_ae]

fig, ax = plt.subplots(figsize=(7, 4))
bars = ax.bar(recon_methods, recon_mses, color=["steelblue", "darkorange", "forestgreen"],
              edgecolor="black")
for bar, val in zip(bars, recon_mses):
    ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.0005,
            f"{val:.4f}", ha="center", va="bottom", fontsize=10)
ax.set_ylabel("Reconstruction MSE")
ax.set_title("Reconstruction Error (Test Set)")
plt.tight_layout()
plt.savefig("reconstruction_mse.png", dpi=150)
plt.close()
print("Saved: reconstruction_mse.png")


# ─────────────────────────────────────────────────────────────────────────────
# 13.  k-NN Downstream Classification Comparison
# ─────────────────────────────────────────────────────────────────────────────

df_results = pd.DataFrame(results)

print(df_results.to_string(index=False))
df_results.to_csv("q3_all_results.csv", index=False)
print("\nSaved: q3_all_results.csv")

# Bar chart: k-NN accuracy for each method variant
# Extract numeric accuracy
def parse_acc(s):
    try:
        return float(str(s).split("±")[0])
    except Exception:
        return np.nan

df_results["acc_val"] = df_results["k-NN Acc."].apply(parse_acc)
df_plot = df_results.dropna(subset=["acc_val"]).sort_values("acc_val", ascending=False)

fig, ax = plt.subplots(figsize=(10, 5))
colors = plt.cm.viridis(np.linspace(0.2, 0.9, len(df_plot)))
bars = ax.barh(df_plot["Method"], df_plot["acc_val"], color=colors, edgecolor="black")
ax.set_xlabel("k-NN Accuracy (5-fold CV)")
ax.set_title("Downstream Classification Accuracy (k-NN, k=5) by Reduction Method")
ax.set_xlim(0, 1)
for bar, val in zip(bars, df_plot["acc_val"]):
    ax.text(val + 0.005, bar.get_y() + bar.get_height() / 2,
            f"{val:.4f}", va="center", fontsize=9)
plt.tight_layout()
plt.savefig("knn_accuracy_comparison.png", dpi=150)
plt.close()
print("Saved: knn_accuracy_comparison.png")