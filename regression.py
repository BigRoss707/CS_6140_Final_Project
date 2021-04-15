import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import LinearRegression
from scipy.stats import pearsonr

case_info = np.genfromtxt('data/combined_data_small_pop.csv', delimiter=',')
X_small = case_info[1:, 6]
Y_small = case_info[1:, 18]

X_small = X_small.reshape(-1, 1)

reg_small = LinearRegression().fit(X_small, Y_small)
score_small = reg_small.score(X_small, Y_small)
Y_predict_small = reg_small.predict(X_small)

plt.scatter(X_small, Y_small, s=4)
plt.plot(X_small, Y_predict_small, color='red')
plt.title('Mask Use and Percent of Cases in Small (<100,000) Counties')
plt.xlabel('Always or Frequently Use Masks')
plt.ylabel('Percent of Cases in County')
plt.show()

# calculate Pearson's correlation
corr_small, _ = pearsonr(X_small[:, 0], Y_small)


case_info_all = np.genfromtxt('data/combined_data_large_pop.csv', delimiter=',')
X_large = case_info_all[1:, 6]
Y_large = case_info_all[1:, 18]

X_large = X_large.reshape(-1, 1)

reg_large = LinearRegression().fit(X_large, Y_large)
score_large = reg_large.score(X_large, Y_large)
Y_predict_large = reg_small.predict(X_large)

plt.scatter(X_large, Y_large, s=4)
plt.plot(X_large, Y_predict_large, color='red')
plt.title('Mask Use and Percent of Cases in Large (>100,000) Counties')
plt.xlabel('Always or Frequently Use Masks')
plt.ylabel('Percent of Cases in County')
plt.show()

# calculate Pearson's correlation
corr_large, _ = pearsonr(X_large[:, 0], Y_large)
