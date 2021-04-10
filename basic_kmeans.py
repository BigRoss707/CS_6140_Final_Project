import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import clusters
import plot

data = clusters.get_data()

num_clusters = 4
pipe = clusters.get_pipeline(num_clusters)
pipe.fit(data)
_, _, data_pca = clusters.get_pca_data(num_clusters)
data_with_clusters = data
data_with_clusters['predicted_cluster'] = pipe.predict(data)

mean_vals = data_with_clusters.groupby('predicted_cluster', as_index=False).mean()
max_vals = data_with_clusters.groupby('predicted_cluster', as_index=False).max()
min_vals = data_with_clusters.groupby('predicted_cluster', as_index=False).min()
std_dev_vals = data_with_clusters.groupby('predicted_cluster', as_index=False).std()

# print(min_vals.to_markdown())

# Finally we plot all of our data and make it look a bit pretty
plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))
sns.scatterplot(
    x="component_1",
    y="component_2",
    s=50,
    data=data_pca,
    hue="predicted_cluster",
    style="predicted_cluster",
    palette="Set2",
)
plt.show()

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))
sns.scatterplot(
    x="component_1",
    y="component_3",
    s=50,
    data=data_pca,
    hue="predicted_cluster",
    style="predicted_cluster",
    palette="Set2",
)
plt.show()

plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))
sns.scatterplot(
    x="component_2",
    y="component_3",
    s=50,
    data=data_pca,
    hue="predicted_cluster",
    style="predicted_cluster",
    palette="Set2",
)
plt.show()


# Create a data frame containing our centroids
plot.display_parallel_coordinates(data_with_clusters, num_clusters)

plot.display_parallel_coordinates_centroids(mean_vals, num_clusters)
