import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import clusters
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

palette = sns.color_palette("bright", 10)

data = clusters.get_data()

scaler = StandardScaler()
data = scaler.fit_transform(data)

pca = PCA(n_components=17, random_state=42)
data_pca = pca.fit(data)

PC_values = np.arange(pca.n_components_) + 1
plt.plot(PC_values, pca.explained_variance_ratio_, 'ro-', linewidth=1)
plt.title('Scree Plot')
plt.xlabel('Principal Component')
plt.ylabel('Proportion of Variance Explained')
plt.show()

eigenvalues = pca.explained_variance_
