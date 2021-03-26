import pandas as pd

# Get cases and death information from New York City
case_info = pd.read_csv('data/boroughs-case-hosp-death.csv', usecols = [1,3,4,6,7,9,10,12,13,15])
# Rename column so that the merge works correctly
case_info_totals = case_info.sum()
