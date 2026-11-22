import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import (KMeans, DBSCAN, AgglomerativeClustering)
from sklearn.metrics import (silhouette_score, davies_bouldin_score, calinski_harabasz_score)
import matplotlib.pyplot as plt
import seaborn as sns
df = pd.read_excel('online_retail_II.xlsx')
df = df.dropna(subset=['CustomerID'])
df = df[df['Quantity'] > 0] 
df = df[df['UnitPrice'] > 0]
df['Revenue'] = df['Quantity'] * df['UnitPrice']
reference_date = pd.to_datetime('2011-12-10') 
rfm = df.groupby('CustomerID').agg(
Recency=('InvoiceDate',lambda x: (reference_date - x.max()).days),
Frequency=('InvoiceNo', 'nunique'),
Monetary=('Revenue', 'sum')
).reset_index()
rfm['AvgOrderValue'] = rfm['Monetary'] / rfm['Frequency']
rfm['ProductDiversity'] = df.groupby('CustomerID')['StockCode'].nunique().values 
rfm['ReturnRate'] = 0.0 
rfm['DaysActive'] = df.groupby('CustomerID')['InvoiceDate'].apply( 
lambda x: (x.max() - x.min()).days 
).values 
rfm['PurchaseRegularity'] = df.groupby('CustomerID')['InvoiceDate'].apply( 
lambda x: x.sort_values().diff().dt.days.std() if len(x) > 1 else 0 
).values.fillna(0) 
features = ['Recency', 'Frequency', 'Monetary', 'AvgOrderValue', 
'ProductDiversity', 'ReturnRate', 'DaysActive', 'PurchaseRegularity'] 
scaler = StandardScaler() 
X_scaled = scaler.fit_transform(rfm[features])
pca = PCA(n_components=4) 
X_pca = pca.fit_transform(X_scaled) 
print(f'Explained variance: {pca.explained_variance_ratio_.sum():.3f}')
kmeans = KMeans(n_clusters=5, init='k-means++', n_init=10, max_iter=300, random_state=42) 
rfm['KMeans_Cluster'] = kmeans.fit_predict(X_pca) 
print(f'K-Means Silhouette: {silhouette_score(X_pca, rfm["KMeans_Cluster"]):.3f}') 
print(f'K-Means Davies-Bouldin: {davies_bouldin_score(X_pca, rfm["KMeans_Cluster"]):.3f}') 
print(f'K-Means             Calinski-Harabasz:             {calinski_harabasz_score(X_pca,
rfm["KMeans_Cluster"]):.1f}')
dbscan = DBSCAN(eps=1.2, min_samples=8) 
rfm['DBSCAN_Cluster'] = dbscan.fit_predict(X_pca)  
n_clusters = len(set(rfm['DBSCAN_Cluster'])) - (1 if -1 in rfm['DBSCAN_Cluster'].values else 
0) 
print(f'DBSCAN Clusters: {n_clusters}') 
noise_pct = (rfm['DBSCAN_Cluster'] == -1).mean() * 100 
print(f'DBSCAN Noise: {noise_pct:.1f}%')
hierarchical = AgglomerativeClustering(n_clusters=5, linkage='ward') 
rfm['Hierarchical_Cluster'] = hierarchical.fit_predict(X_pca) 
print(f'Hierarchical Silhouette: {silhouette_score(X_pca, rfm["Hierarchical_Cluster"]):.3f}')
segment_profiles = rfm.groupby('KMeans_Cluster')[features].mean()
print(segment_profiles)
inertia = [] 
K_range = range(2, 11)
for k in K_range: 
    km = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
km.fit(X_pca) 
inertia.append(km.inertia_) 
plt.figure(figsize=(8, 5)) 
plt.plot(K_range, inertia, 'bo-') 
plt.xlabel('Number of Clusters (K)') 
plt.ylabel('Within-Cluster Sum of Squares') 
plt.title('Elbow Method for Optimal K') 
plt.savefig('elbow_method.png') 
plt.show()
