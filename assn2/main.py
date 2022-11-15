import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


#plotting data
data = pd.read_csv('data.csv')
data.columns=['x','y']
plt.scatter(data.x, data.y)
plt.title('visualisaiton of data')
plt.show()



#first we scale the data
scaler = StandardScaler()

data = scaler.fit_transform(data[['x','y']])
data = pd.DataFrame(data)

plt.scatter(data.iloc[:,0], data.iloc[:,1])
plt.title('visualisaiton of scaled data')
plt.show()


#KMEANS
kmeans_clusters = KMeans(n_clusters=2, n_init=20, random_state=23).fit(data)
plt.scatter(data.iloc[:,0], data.iloc[:,1], c=kmeans_clusters.labels_)
plt.title('classification using KMEANS, n_clusters=2')
plt.show()

#DBSCAN
clusters = DBSCAN(eps=0.2, min_samples=30).fit(data)
plt.scatter(data.iloc[:,0], data.iloc[:,1], c=clusters.labels_)
plt.title('classification using DBSCAN (e=0.2, r=30)')
plt.show()

clusters = DBSCAN(eps=0.2, min_samples=50).fit(data)
plt.scatter(data.iloc[:,0], data.iloc[:,1], c=clusters.labels_)
plt.title('classification using DBSCAN (e=0.1, r=40)')
plt.show()


clusters = DBSCAN(eps=0.31, min_samples=80).fit(data)
plt.scatter(data.iloc[:,0], data.iloc[:,1], c=clusters.labels_)
plt.title('classification using DBSCAN (e=0.31, n=80)')
unique, counts = np.unique(clusters.labels_, return_counts=True)
#print(np.asarray((unique, counts)).T)
plt.show()


output = open('output.txt','w')

for label in clusters.labels_:
    output.writelines(str(label)+'\n')

output.close()