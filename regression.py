import matplotlib.pyplot as plt
import numpy as np

case_info = np.genfromtxt('data/combined_data_small_pop.csv', delimiter=',')
plt.scatter(case_info[:, 6], case_info[:, 19], s=4)
plt.title('Mask Use and Percent of Deaths in Small (<100,000) Counties')
plt.xlabel('Always or Frequently Use Masks')
plt.ylabel('Percent of Cases in County')
plt.show()

case_info_all = np.genfromtxt('data/combined_data_large_pop.csv', delimiter=',')
plt.scatter(case_info_all[:, 6], case_info_all[:, 19], s=4)
plt.title('Mask Use and Percent of Deaths in Large (>100,000) Counties')
plt.xlabel('Always or Frequently Use Masks')
plt.ylabel('Percent of Cases in County')
plt.show()

#plt.scatter(case_info_all[:, 5], case_info_all[:, 16])
#plt.xlabel('Always Use Masks')
#plt.ylabel('Percent of Cases in County')