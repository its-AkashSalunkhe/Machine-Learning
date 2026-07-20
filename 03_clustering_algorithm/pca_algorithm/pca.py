from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
import pandas as pd

dataset = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\clustering_algorithm\k-means_algorithm\Mall_Customers.csv")
x = dataset.iloc[:, 2:5]

# Instantiate the PCA model
pca = PCA(n_components=2)

# Fit and transform the data
X_pca = pca.fit_transform(x)


# Create DataFrame for plotting
pca_df = pd.DataFrame(X_pca, columns=['PC1', 'PC2'])


# Visualize the reduced-dimensional data
sns.scatterplot(data=pca_df, x='PC1', y='PC2', s=70)
plt.title('PCA of Mall Customers Dataset')
plt.xlabel('Principal Component 1')
plt.ylabel('Principal Component 2')
plt.legend()
plt.show()





#########################            OR            ###############################





from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

# Load dataset
dataset = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\clustering_algorithm\k-means_algorithm\Mall_Customers.csv")

# Features
x = dataset.iloc[:, 2:5]      # Age, Annual Income, Spending Score

# Labels for coloring
y = dataset["Genre"]           # In dataset Genre = Gender

# PCA
pca = PCA(n_components=2)
X_pca = pca.fit_transform(x)

# Plot
plt.figure(figsize=(8,6))
sns.scatterplot(
    x=X_pca[:,0],
    y=X_pca[:,1],
    hue=y,
    palette="viridis",
    s=70
)

plt.title("PCA of Mall Customers Dataset")
plt.xlabel("Principal Component 1")
plt.ylabel("Principal Component 2")
plt.show()