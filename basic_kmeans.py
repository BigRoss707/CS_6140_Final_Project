import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
case_info = pd.read_csv('data/us-counties-covid-death-July.csv')
# Rename column so that the merge works correctly
case_info = case_info.rename({'fips': 'FIPS Code'}, axis='columns')

# Now sum to get the total number of cases and deaths for each county
case_info_totals = case_info.groupby(['FIPS Code']).sum()

# Merge all data with mask use data by the County FIPS code
data = mask_use.merge(county_education, on='FIPS Code', how='inner')
data = data.merge(population_estimation, on='FIPS Code', how='inner')
data = data.merge(unemployment, on='FIPS Code', how='inner')
data = data.merge(case_info_totals, on='FIPS Code', how='inner')
data = data.drop('FIPS Code', axis=1)

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
                n_clusters=4,
                init="k-means++",
                random_state=42,
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

# Now we run all steps on our data set
pipe.fit(data)

# Now we collect all results with the most important components from PCA
pcadf = pd.DataFrame(
    pipe["preprocessor"].transform(data),
    columns=["component_1", "component_2", "component_3"],
)

# Now we get the predicted value from each instance
pcadf["predicted_cluster"] = pipe["cluster"]["kmeans"].labels_

# Finally we plot all of our data and make it look a bit pretty
plt.style.use("fivethirtyeight")
plt.figure(figsize=(8, 8))
sns.scatterplot(
    x="component_1",
    y="component_2",
    s=50,
    data=pcadf,
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
    data=pcadf,
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
    data=pcadf,
    hue="predicted_cluster",
    style="predicted_cluster",
    palette="Set2",
)
plt.show()
