
import numpy as np
import pandas as pd
import scipy
from scipy.optimize import linprog



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
case_info = pd.read_csv('us-counties-covid-death-July.csv')
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

# Turn Data into a numpy array

data_array = data.to_numpy()



#Create an empty list to hold the Data Envelopment Scores.

dea_scores = []




# Class that takes in a data set and has a method
# that returns the efficiency score of a given row.
class DEA:
    def __init__(self,data):
        self.data = data

    def get_coef_matrix(self):
        matrix = []
        for i in range(self.data.shape[0]):
            new_row = []
            for j in range(self.data.shape[1]-2):
                x = -self.data[i,j]
                new_row.append(x)
            new_row.append(self.data[i,14])
            new_row.append(self.data[i,15])
            matrix.append(new_row)
        new_matrix = np.array(matrix)
        return new_matrix



    def compute_score(self,row):
        c_vec = []
        for i in range(self.data.shape[1]-2):
            c_vec.append(self.data[row,i])
        c_vec.append(-self.data[row,14])
        c_vec.append(-self.data[row,15])
        c_vec = np.array(c_vec)



        A_matrix = self.get_coef_matrix()



        v_matrix = []
        for i in range(self.data.shape[1]-2):
            v_matrix.append(self.data[row,i])
        v_matrix.append(0)
        v_matrix.append(0)
        v_matrix = np.array(v_matrix)
        v_matrix =v_matrix.reshape((1,16))



        zero_vec = [0 for i in range(self.data.shape[0])]
        zero_vec = np.array(zero_vec)

        one_vec = [1]
        one_vec = np.array(one_vec)

        res = scipy.optimize.linprog(c_vec, A_ub = A_matrix, b_ub = zero_vec, A_eq = v_matrix, b_eq = one_vec )
        weights = res.x

        a_1 = self.data[row,-1]
        a_2 = self.data[row,-2]
        u_1 = weights[-1]
        u_2 = weights[-2]

        score = a_1*u_1 + a_2*u_2

        return score


#create an instance of the DEA class using the data set we have above

dea_instance = DEA(data_array)


# Loop through the data to get a score for each row.

for i in range(data.shape[0]):
    score = dea_instance.compute_score(i)
    dea_scores.append(score)
    if i%10 == 0:
        print(score)

dea_scores = np.array(dea_scores)
dea_scores = dea_scores.transpose()

np.savetxt('dea_scores.csv',dea_scores)








