import pandas as pd
from sklearn.cluster import Birch, DBSCAN, SpectralClustering, AgglomerativeClustering, KMeans
from sklearn.metrics import normalized_mutual_info_score
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler


X = pd.read_csv('data.csv', header=None)

plt.scatter(X.iloc[:,0],X.iloc[:,1])
plt.show()

#scaling the data
scaler = StandardScaler()

X = scaler.fit_transform(X)
X = pd.DataFrame(X)

clf = KMeans(n_clusters=2)
labels = clf.fit(X).labels_
plt.scatter(X.iloc[:,0],X.iloc[:,1], c=labels)
plt.title('KMeans')
plt.show()


clf = Birch(n_clusters=2, threshold = 0.3)
labels = clf.fit(X).labels_
plt.scatter(X.iloc[:,0],X.iloc[:,1], c=labels)
plt.title('Birch')
plt.show()

clf = SpectralClustering(n_clusters=2)
labels = clf.fit(X).labels_
plt.scatter(X.iloc[:,0],X.iloc[:,1], c=labels)
plt.title('Spectral')
plt.show()

clf = AgglomerativeClustering(n_clusters=2)
labels = clf.fit(X).labels_
plt.scatter(X.iloc[:,0],X.iloc[:,1], c=labels)
plt.title('Agglomerative Clustering')
plt.show()

clf = DBSCAN(eps=0.31, min_samples=80)
labels = clf.fit(X).labels_
plt.scatter(X.iloc[:,0],X.iloc[:,1], c=labels)
plt.title('DBScan')
plt.show()


file = open('output.txt','w')
for label in labels:
    file.write(str(label)+'\n')
file.close()