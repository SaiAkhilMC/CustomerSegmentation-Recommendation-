#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Import necessary libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA


# In[5]:


# Load the dataset
data = pd.read_csv('data.csv', encoding='ISO-8859-1')


# In[6]:


# Display the first few rows of the dataset
print("First few rows of the dataset:")
print(data.head())


# In[7]:


# Perform EDA
print("\nBasic Statistics:")
print(data.describe())

print("\nMissing Values:")
print(data.isnull().sum())


# In[8]:


# Visualize data distributions
plt.figure(figsize=(10, 6))
sns.pairplot(data)
plt.title('Pairplot of the dataset')
plt.show()


# In[10]:


# Identify non-numeric columns
non_numeric_columns = data.select_dtypes(include=['object']).columns
print("Non-numeric columns:", non_numeric_columns)

# Drop non-numeric columns for clustering purposes
data_numeric = data.drop(columns=non_numeric_columns)

# Check if there are still any missing values
print("\nMissing Values in Numeric Data:")
print(data_numeric.isnull().sum())

# Drop rows with missing values in numeric data
data_numeric = data_numeric.dropna()

# Standardize the numeric data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Display the shape of the scaled data
print("\nShape of scaled data:", data_scaled.shape)


# In[11]:


# Perform PCA for visualization
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 6))
plt.scatter(data_pca[:, 0], data_pca[:, 1], s=50, cmap='viridis')
plt.title('PCA of the dataset')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.show()


# In[16]:


# Drop rows with missing values in numeric data
data_numeric = data_numeric.dropna()

# Synchronize the original data by dropping the same rows with missing values
data_synchronized = data.dropna(subset=data_numeric.columns)

# Ensure the indexes are aligned
assert data_synchronized.index.equals(data_numeric.index), "Indexes are not aligned!"

# Standardize the numeric data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters as needed
clusters = kmeans.fit_predict(data_scaled)

# Add cluster labels to the synchronized original data
data_synchronized['Cluster'] = clusters


# In[18]:


# Visualize the clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=clusters, palette='viridis')
plt.title('Customer Segments')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend(title='Cluster')
plt.show()

# Select only the numeric columns
numeric_columns = data_synchronized.select_dtypes(include=[np.number]).columns

# Calculate and display cluster statistics for numeric columns
cluster_statistics = data_synchronized.groupby('Cluster')[numeric_columns].mean()

print("\nCluster Statistics:")
print(cluster_statistics)


# In[21]:


import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

# Drop rows with missing values in numeric data
data_numeric = data.drop(columns=non_numeric_columns).dropna()

# Synchronize the original data by dropping the same rows with missing values
data_synchronized = data.dropna(subset=data_numeric.columns)

# Ensure the indexes are aligned
assert data_synchronized.index.equals(data_numeric.index), "Indexes are not aligned!"

# Standardize the numeric data
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data_numeric)

# Perform K-Means clustering
kmeans = KMeans(n_clusters=3, random_state=42)  # Adjust the number of clusters as needed
clusters = kmeans.fit_predict(data_scaled)

# Add cluster labels to the synchronized original data
data_synchronized['Cluster'] = clusters

# Visualize the clusters using PCA for dimensionality reduction
pca = PCA(n_components=2)
data_pca = pca.fit_transform(data_scaled)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=data_pca[:, 0], y=data_pca[:, 1], hue=clusters, palette='viridis')
plt.title('Customer Segments')
plt.xlabel('First Principal Component')
plt.ylabel('Second Principal Component')
plt.legend(title='Cluster')
plt.show()

# Select only the numeric columns
numeric_columns = data_synchronized.select_dtypes(include=[np.number]).columns

# Calculate and display cluster statistics for numeric columns
cluster_statistics = data_synchronized.groupby('Cluster')[numeric_columns].mean()

print("\nCluster Statistics:")
print(cluster_statistics)


# In[ ]:




