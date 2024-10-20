import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Set random seed for reproducibility
np.random.seed(42)

# Define the number of samples for each rock type
n_samples = 300


# Function to generate correlated data
def generate_correlated_data(mean, cov, n):
    return np.random.multivariate_normal(mean, cov, n)


# Generate synthetic data for igneous rocks
igneous_mean = [70, 14, 3, 2, 1, 3, 5, 0.3]
igneous_cov = np.array(
    [
        [25, -5, -2, -1, -0.5, 1, 2, -0.1],
        [-5, 4, 0.5, 0.2, 0.1, -0.2, -0.4, 0.02],
        [-2, 0.5, 1, 0.1, 0.05, -0.1, -0.2, 0.01],
        [-1, 0.2, 0.1, 0.5, 0.02, -0.05, -0.1, 0.005],
        [-0.5, 0.1, 0.05, 0.02, 0.2, -0.02, -0.05, 0.002],
        [1, -0.2, -0.1, -0.05, -0.02, 0.5, 0.1, -0.005],
        [2, -0.4, -0.2, -0.1, -0.05, 0.1, 1, -0.01],
        [-0.1, 0.02, 0.01, 0.005, 0.002, -0.005, -0.01, 0.05],
    ]
)
igneous_data = generate_correlated_data(igneous_mean, igneous_cov, n_samples)

# Generate synthetic data for sedimentary rocks
sedimentary_mean = [50, 10, 5, 20, 5, 1, 2, 0.5]
sedimentary_cov = np.array(
    [
        [100, -10, -5, -20, -5, -1, -2, -0.5],
        [-10, 9, 1, 2, 0.5, 0.1, 0.2, 0.05],
        [-5, 1, 4, 1, 0.25, 0.05, 0.1, 0.025],
        [-20, 2, 1, 36, 2, 0.2, 0.4, 0.1],
        [-5, 0.5, 0.25, 2, 4, 0.05, 0.1, 0.025],
        [-1, 0.1, 0.05, 0.2, 0.05, 0.1, 0.02, 0.005],
        [-2, 0.2, 0.1, 0.4, 0.1, 0.02, 0.4, 0.01],
        [-0.5, 0.05, 0.025, 0.1, 0.025, 0.005, 0.01, 0.04],
    ]
)
sedimentary_data = generate_correlated_data(sedimentary_mean, sedimentary_cov, n_samples)

# Generate synthetic data for metamorphic rocks
metamorphic_mean = [60, 18, 8, 8, 4, 2, 2, 0.8]
metamorphic_cov = np.array(
    [
        [64, -12, -6, -4, -2, -1, -1, -0.4],
        [-12, 7, 1, 0.5, 0.25, 0.125, 0.125, 0.05],
        [-6, 1, 4, 0.25, 0.125, 0.0625, 0.0625, 0.025],
        [-4, 0.5, 0.25, 7, 0.125, 0.0625, 0.0625, 0.025],
        [-2, 0.25, 0.125, 0.125, 2, 0.03125, 0.03125, 0.0125],
        [-1, 0.125, 0.0625, 0.0625, 0.03125, 0.5, 0.015625, 0.00625],
        [-1, 0.125, 0.0625, 0.0625, 0.03125, 0.015625, 0.5, 0.00625],
        [-0.4, 0.05, 0.025, 0.025, 0.0125, 0.00625, 0.00625, 0.1],
    ]
)
metamorphic_data = generate_correlated_data(metamorphic_mean, metamorphic_cov, n_samples)

# Combine all rock types
components = ["SiO2", "Al2O3", "Fe2O3", "CaO", "MgO", "Na2O", "K2O", "TiO2"]
igneous = pd.DataFrame(igneous_data, columns=components)
igneous["rock_type"] = "igneous"
sedimentary = pd.DataFrame(sedimentary_data, columns=components)
sedimentary["rock_type"] = "sedimentary"
metamorphic = pd.DataFrame(metamorphic_data, columns=components)
metamorphic["rock_type"] = "metamorphic"

rocks = pd.concat([igneous, sedimentary, metamorphic], ignore_index=True)

# Shuffle the dataset
rocks = rocks.sample(frac=1).reset_index(drop=True)

# Save the dataset
rocks_type = rocks["rock_type"]
rocks_type.to_csv("data/rock_types.csv", index=False)
rocks_sample = rocks.drop(columns="rock_type")
rocks_sample.to_csv("data/rock_samples.csv", index=False)
# rocks.to_csv("data/rock_samples.csv", index=False)

print("Synthetic rock sample dataset has been created and saved to 'data/rock_samples.csv'")

# Visualize the data
plt.figure(figsize=(12, 10))
sns.pairplot(rocks, hue="rock_type", vars=components, plot_kws={"alpha": 0.5})
plt.tight_layout()
# plt.show()
plt.savefig("pairplot.png")
plt.close()

# Perform PCA
X = rocks[components]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=rocks["rock_type"].astype("category").cat.codes, alpha=0.6)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("PCA of Rock Samples")
plt.colorbar(scatter, label="Rock Type")
# plt.show()
plt.savefig("pca_plot.png")
plt.close()

# Perform K-means clustering
kmeans = KMeans(n_clusters=3, random_state=19)
kmeans_labels = kmeans.fit_predict(X_scaled)

plt.figure(figsize=(10, 8))
scatter = plt.scatter(X_pca[:, 0], X_pca[:, 1], c=kmeans_labels, alpha=0.6)
plt.xlabel("First Principal Component")
plt.ylabel("Second Principal Component")
plt.title("K-means Clustering of Rock Samples")
plt.colorbar(scatter, label="Cluster")
# plt.show()
plt.savefig("kmeans_plot.png")
plt.close()

print("Visualizations have been saved as 'pairplot.png', 'pca_plot.png', and 'kmeans_plot.png'")
