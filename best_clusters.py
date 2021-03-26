import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

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

def plot_elbow_method(data):
    distortions = []
    for i in range(1, 25):
        pipe = get_pipeline(i)

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
        pipe = get_pipeline(n_clusters)
        pipe.fit(data)

        # Now we get the predicted value from each instance
        labels = pipe["cluster"]["kmeans"].labels_
        if usePCA:
            pcaData = get_pca_data(data, pipe)
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

# Read in the data
county_education = pd.read_excel('data/Education.xls', skiprows=(0, 1, 2, 3), usecols=(0, 5, 6, 43, 44, 45, 46))
county_education = county_education.rename({'Percent of adults with less than a high school diploma, 2014-18': 'Per Less than HS',
                                            'Percent of adults with a high school diploma only, 2014-18': 'Per Only HS',
                                            'Percent of adults completing some college or associate\'s degree, 2014-18': 'Per Some College',
                                            'Percent of adults with a bachelor\'s degree or higher, 2014-18': 'Per Bachelors or Higher',
                                            '2013 Rural-urban Continuum Code': 'Rural Code',
                                            '2013 Urban Influence Code': 'Urban Code'}, axis='columns')
county_education['Per Less than HS'] = county_education['Per Less than HS']/100
county_education['Per Only HS'] = county_education['Per Only HS']/100
county_education['Per Some College'] = county_education['Per Some College']/100
county_education['Per Bachelors or Higher'] = county_education['Per Bachelors or Higher']/100

# Column 0 is the FIPS code, 19 is the most recent population estimate
population_estimation = pd.read_excel('data/PopulationEstimates.xls', skiprows=(0, 1), usecols=(0, 19))
# Rename column so that the merge works correctly
population_estimation = population_estimation.rename({'FIPStxt': 'FIPS Code', 'POP_ESTIMATE_2019': 'Population'}, axis='columns')

# Column 0 is the FIPS code
# 85 is the most recent unemployment rate, 86 is the percent of median income compared to average
unemployment = pd.read_excel('data/Unemployment.xls', skiprows=(0, 1, 2, 3), usecols=(0, 85, 87))
# Rename column so that the merge works correctly
unemployment = unemployment.rename({'fips_txt': 'FIPS Code', 'Unemployment_rate_2019': 'Unemployment Rate',
                                    'Med_HH_Income_Percent_of_State_Total_2019': 'Per of Median HH Income'}, axis='columns')
# Convert to values between 0 and 1
unemployment['Unemployment Rate'] = unemployment['Unemployment Rate']/100
unemployment['Per of Median HH Income'] = unemployment['Per of Median HH Income']/100

# Column 0 is the FIPS code
# 10 is percent of people in poverty
poverty = pd.read_excel('data/PovertyEstimates.xls', skiprows=(0, 1, 2, 3), usecols=(0, 10))
# Rename column so that the merge works correctly
poverty = poverty.rename({'FIPStxt': 'FIPS Code', 'PCTPOVALL_2019': 'Per in Poverty'}, axis='columns')
poverty['Per in Poverty'] = poverty['Per in Poverty']/100

# Get mask use data
mask_use = pd.read_csv('data/mask-use-by-county.csv')
# Rename column so that the merge works correctly
mask_use = mask_use.rename({'FIPS': 'FIPS Code', 'NEVER': 'Never', 'RARELY': 'Rarely', 'SOMETIMES': 'Sometimes',
                            'FREQUENTLY': 'Frequently', 'ALWAYS': 'Always'}, axis='columns')

# Get cases and death information
case_info = pd.read_csv('data/us-counties-covid-death-on-August-1.csv', usecols=(2, 3, 4))
# Rename column so that the merge works correctly
case_info = case_info.rename({'fips': 'FIPS Code', 'cases': 'Per of Cases', 'deaths': 'Per of Deaths'}, axis='columns')

# Merge all data with mask use data by the County FIPS code
data = mask_use.merge(county_education, on='FIPS Code', how='inner')
data = data.merge(population_estimation, on='FIPS Code', how='inner')
data = data.merge(unemployment, on='FIPS Code', how='inner')
data = data.merge(poverty, on='FIPS Code', how='inner')
data = data.merge(case_info, on='FIPS Code', how='inner')
data = data.drop('FIPS Code', axis=1)

data['Per of Cases'] = data['Per of Cases']/data['Population']
data['Per of Deaths'] = data['Per of Deaths']/data['Population']

plot_elbow_method(data)
plot_average_silhouette(data,40)
plot_average_silhouette(data,8,usePCA=False)
