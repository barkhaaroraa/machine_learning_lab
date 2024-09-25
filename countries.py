# Step 2: Importing Required Libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.cluster import KMeans

# Set seaborn default style for better visuals
# sns.set()

# Step 3: Loading the Data
raw_data = pd.read_csv('C:/Users/barkha arora/Desktop/machine learning/countries/Countries-exercise.csv')
print(raw_data.head())  # Display the first few rows of the data

# Step 4: Copying the Data
data = raw_data.copy()

# Step 5: Visualizing the Geographical Data
plt.scatter(data['Longitude'], data['Latitude'])
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.title('Geographical Location of Countries')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# Step 6: Selecting Features for Clustering
x = data.iloc[:, 1:3]  # Select longitude and latitude columns

# Step 7: Initializing the K-Means Algorithm
kmeans = KMeans(n_clusters=7)

# Step 8: Training the K-Means Algorithm
kmeans.fit(x)

# Step 9: Making Predictions with K-Means
identified_clusters = kmeans.fit_predict(x)

# Step 10: Adding Cluster Information to the DataFrame
data_with_clusters = data.copy()
data_with_clusters['Cluster'] = identified_clusters

# Step 11: Visualizing the Clusters
plt.scatter(data['Longitude'], data['Latitude'], c=data_with_clusters['Cluster'], cmap='rainbow')
plt.xlim(-180, 180)
plt.ylim(-90, 90)
plt.title('Clusters of Countries')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()