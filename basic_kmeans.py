import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from pandas.plotting import parallel_coordinates
import plotly.express as px
import plotly.io as pio

def get_pipeline(clusters):
    # Create the preprocessor step
    # Preprocessing will scale all data appropriately since the column values have different ranges and scales
    # dimensionality reduction step to reduce the data into important
    # components using PCA
    preprocessor = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("pca", PCA(n_components=3, random_state=42)),
        ]
    )

    # The cluster step in the pipeline will run kmeans clustering
    cluster = Pipeline(
        [
            (
                "kmeans",
                KMeans(
                    n_clusters=clusters,
                    init="k-means++",
                    random_state=42
                ),
            ),
        ]
    )

    # The pipeline creates an easy way for us to run all steps in Sklearn
    # We can just fit the data to the pipeline and it will run the preprocessing
    # step and then run the clustering algorithm
    pipe = Pipeline(
        [
            ("preprocessor", preprocessor),
            ("cluster", cluster)
        ]
    )
    return pipe

def get_pca_data(data, pipe):
    pcadf = pd.DataFrame(
        pipe["preprocessor"].transform(data),
        columns=["component_1", "component_2", "component_3"],
    )

    # Now we get the predicted value from each instance
    pcadf["predicted_cluster"] = pipe["cluster"]["kmeans"].labels_

    return pcadf

def get_centers(pipe):
    centers = pipe["cluster"]["kmeans"].cluster_centers_
    return centers

# Some of this code modified from https://github.com/OpenClassrooms-Student-Center/Multivariate-Exploratory-Analysis
def addAlpha(colour, alpha):
    '''Add an alpha to the RGB colour'''
    return (colour[0], colour[1], colour[2], alpha)

def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Create the plot
    fig = plt.figure(figsize=(12, 5))
    title = fig.suptitle("Parallel Coordinates plot for the Centroids", fontsize=18)
    fig.subplots_adjust(top=0.9, wspace=0)

    # Draw the chart
    parallel_coordinates(df, 'predicted_cluster', color=palette)

    # Stagger the axes
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

def display_parallel_coordinates(df, num_clusters):
    '''Display a parallel coordinates plot for the clusters in df'''

    # Select data points for individual clusters
    cluster_points = []
    for i in range(num_clusters):
        cluster_points.append(df[df.predicted_cluster == i])

    # Draw the chart
    pc = px.parallel_coordinates(data_frame=df, color='predicted_cluster')
    pio.renderers.default = 'browser'
    pc.show()

    # Stagger the axes
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

def display_parallel_coordinates_centroids(df, num_clusters):
    '''Display a parallel coordinates plot for the centroids in df'''

    # Draw the chart
    pc = px.parallel_coordinates(data_frame=df, color='predicted_cluster')
    pio.renderers.default = 'browser'
    pc.show()

    # Stagger the axes
    ax = plt.gca()
    for tick in ax.xaxis.get_major_ticks()[1::2]:
        tick.set_pad(20)

palette = sns.color_palette("bright", 10)

# Read in the data
county_education = pd.read_excel('data/Education.xls', skiprows=(0, 1, 2, 3), usecols=(0, 5, 6, 43, 44, 45, 46))

# Column 0 is the FIPS code, 19 is the most recent population estimate
population_estimation = pd.read_excel('data/PopulationEstimates.xls', skiprows=(0, 1), usecols=(0, 19))
# Rename column so that the merge works correctly
population_estimation = population_estimation.rename({'FIPStxt': 'FIPS Code'}, axis='columns')

# Column 0 is the FIPS code
# 85 is the most recent unemployment rate, 86 is the percent of median income compared to average
unemployment = pd.read_excel('data/Unemployment.xls', skiprows=(0, 1, 2, 3), usecols=(0, 85, 87))
# Rename column so that the merge works correctly
unemployment = unemployment.rename({'fips_txt': 'FIPS Code'}, axis='columns')

# Column 0 is the FIPS code
# 10 is percent of people in poverty
poverty = pd.read_excel('data/PovertyEstimates.xls', skiprows=(0, 1, 2, 3), usecols=(0, 10))
# Rename column so that the merge works correctly
poverty = poverty.rename({'FIPStxt': 'FIPS Code'}, axis='columns')

# Get mask use data
mask_use = pd.read_csv('data/mask-use-by-county.csv')
# Rename column so that the merge works correctly
mask_use = mask_use.rename({'FIPS': 'FIPS Code'}, axis='columns')

# Get cases and death information
case_info = pd.read_csv('data/us-counties-covid-death-on-August-1.csv', usecols=(2, 3, 4))
# Rename column so that the merge works correctly
case_info = case_info.rename({'fips': 'FIPS Code'}, axis='columns')

# Merge all data with mask use data by the County FIPS code
data = mask_use.merge(county_education, on='FIPS Code', how='inner')
data = data.merge(population_estimation, on='FIPS Code', how='inner')
data = data.merge(unemployment, on='FIPS Code', how='inner')
data = data.merge(case_info, on='FIPS Code', how='inner')
data = data.drop('FIPS Code', axis=1)

data['cases'] = data['cases']/data['POP_ESTIMATE_2019']
data['deaths'] = data['deaths']/data['POP_ESTIMATE_2019']
data = data.drop('POP_ESTIMATE_2019', axis=1)


num_clusters = 4
pipe = get_pipeline(num_clusters)
pipe.fit(data)
data_pca = get_pca_data(data, pipe)
data_with_clusters = data
data_with_clusters['predicted_cluster'] = pipe.predict(data)

mean_vals = data_with_clusters.groupby('predicted_cluster', as_index=False).mean()
max_vals = data_with_clusters.groupby('predicted_cluster', as_index=False).max()
min_vals = data_with_clusters.groupby('predicted_cluster', as_index=False).min()
std_dev_vals = data_with_clusters.groupby('predicted_cluster', as_index=False).std()

# print(mean_vals.to_markdown())


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
display_parallel_coordinates(data_with_clusters, num_clusters)

display_parallel_coordinates_centroids(mean_vals, num_clusters)

