import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score
from clusters import get_data
from clusters import get_pipeline_kmeans
from clusters import get_pca_data_kmeans

def plot_elbow_method(data):
    distortions = []
    for i in range(1, 25):
        pipe = get_pipeline_kmeans(i)

        # Now we run all steps on our data set
        pipe.fit(data)

        # Now we get the predicted value from each instance
        inert = pipe["cluster"]["kmeans"].inertia_

        distortions.append(inert)

    # plot
    plt.plot(range(1, 25), distortions, marker='o')
    plt.xlabel('Number of clusters')
    plt.ylabel('Distortion')
    plt.show()

### A lot of this comes from https://scikit-learn.org/stable/auto_examples/cluster/plot_kmeans_silhouette_analysis.html
### Cite if we need to but we're using their libraries so I don't know if it is necessary
def plot_average_silhouette(data,clusters,usePCA=True):
    for n_clusters in range(2,clusters):

        # Create a subplot with 1 row and 1 columns
        fig, (ax1) = plt.subplots(1, 1)
        fig.set_size_inches(9, 7)

        # The 1st subplot is the silhouette plot
        # The silhouette coefficient can range from -1, 1
        ax1.set_xlim([-1, 1])
        # The (n_clusters+1)*10 is for inserting blank space between silhouette
        # plots of individual clusters, to demarcate them clearly.
        ax1.set_ylim([0, len(data) + (n_clusters + 1) * 10])

        # Initialize the clusterer with n_clusters value and a random generator
        # seed of 10 for reproducibility.
        pipe = get_pipeline_kmeans(n_clusters)
        pipe.fit(data)

        # Now we get the predicted value from each instance
        labels = pipe["cluster"]["kmeans"].labels_
        if usePCA:
            data, pipe, pcaData = get_pca_data_kmeans(n_clusters)
            data = pcaData

        #print(labels)
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(data, labels)
        print("For n_clusters =", n_clusters,
              "The average silhouette_score is :", silhouette_avg)

        # Compute the silhouette scores for each sample
        sample_silhouette_values = silhouette_samples(data, labels)

        y_lower = 10
        for i in range(n_clusters):
            # Aggregate the silhouette scores for samples belonging to
            # cluster i, and sort them
            ith_cluster_silhouette_values = \
                sample_silhouette_values[labels == i]

            ith_cluster_silhouette_values.sort()

            size_cluster_i = ith_cluster_silhouette_values.shape[0]
            y_upper = y_lower + size_cluster_i

            color = cm.nipy_spectral(float(i) / n_clusters)
            ax1.fill_betweenx(np.arange(y_lower, y_upper),
                              0, ith_cluster_silhouette_values,
                              facecolor=color, edgecolor=color, alpha=0.7)

            # Label the silhouette plots with their cluster numbers at the middle
            ax1.text(-0.05, y_lower + 0.5 * size_cluster_i, str(i))

            # Compute the new y_lower for next plot
            y_lower = y_upper + 10  # 10 for the 0 samples

        ax1.set_title("The silhouette plot for the various clusters.")
        ax1.set_xlabel("The silhouette coefficient values")
        ax1.set_ylabel("Cluster label")

         # The vertical line for average silhouette score of all the values
        ax1.axvline(x=silhouette_avg, color="red", linestyle="--")

        ax1.set_yticks([])  # Clear the yaxis labels / ticks
        ax1.set_xticks([-1, -0.8, -.6, -.4, -.2 , 0, 0.2, 0.4, 0.6, 0.8, 1])

    plt.show()

data = get_data()
plot_elbow_method(data)
plot_average_silhouette(data,8)
plot_average_silhouette(data,8,usePCA=False)
