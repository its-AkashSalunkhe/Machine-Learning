# Hierarchial Clustering

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df = pd.read_csv(r"C:\Users\Akash\Desktop\FSDS\ML\clustering_algorithm\hierarchical_clustering\Mall_Customers.csv")

df

x = df.iloc[:,2:5].values

# Using the dendrogram to find the optimal number of clusters
import scipy.cluster.hierarchy as sch

dendogram = sch.dendrogram(sch.linkage(x, method = 'ward'))


#we are going to build the dendogram with only one line of code 
#linkage is one of the hieararchicla clustering algorithm & you have to build the linkage on X 
#ward method actually try to minimise the variance on each cluster & in k-means we minimise the sum of squared 

plt.title("Dendogram")
plt.ylabel("Customers")
plt.xlabel("Euclidean Distance")
plt.plot()


# YOU CAN IMPLETE HEAR FIND ELBOW METOD

# Training the Hierarchical Clustering model on the dataset

from sklearn.cluster import AgglomerativeClustering
classifier = AgglomerativeClustering(n_clusters=5, metric='euclidean', linkage='ward')
y_hc = classifier.fit_predict(x)





# Visualising the clusters
plt.scatter(x[y_hc == 0, 0], x[y_hc == 0, 1], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(x[y_hc == 1, 0], x[y_hc == 1, 1], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(x[y_hc == 2, 0], x[y_hc == 2, 1], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(x[y_hc == 3, 0], x[y_hc == 3, 1], s = 100, c = 'cyan', label = 'Cluster 4')
plt.scatter(x[y_hc == 4, 0], x[y_hc == 4, 1], s = 100, c = 'magenta', label = 'Cluster 5')
plt.title('Clusters of customers')
plt.xlabel('Annual Income (k$)')
plt.ylabel('Spending Score (1-100)')
plt.legend()
plt.show() 



# PLEASE COMPARE BOTH K-MEANS CLUSTERING vs HIERARCHICAL CLUSTERING
df['cluster'] = y_hc


