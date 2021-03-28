import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.cm as cm
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_samples, silhouette_score

##Returns the data from the fileset merged together
def get_data():
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

    return data

##YOU STILL NEED TO FIT THE DATA TO THE PIPELINE
##Returns the pipeline for the data with a number of clusters = num_clusters
def get_pipeline(num_clusters):
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
                    n_clusters=num_clusters,
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

def get_pca_data(num_clusters):
    data = get_data()
    pipe = get_pipeline(num_clusters)
    pipe.fit(data)
    
    pcadf = pd.DataFrame(
        pipe["preprocessor"].transform(data),
        columns=["component_1", "component_2", "component_3"],
    )

    # Now we get the predicted value from each instance
    pcadf["predicted_cluster"] = pipe["cluster"]["kmeans"].labels_

    return data, pipe, pcadf

    
