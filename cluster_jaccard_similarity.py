import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from clusters import labels_kmeans
from clusters import labels_hierarchical

num_clusters = 3
kLabels = labels_kmeans(num_clusters)
hLabels = labels_hierarchical(num_clusters)

print(kLabels)
for i in range(num_clusters):
    for j in range(num_clusters):
        numerator = []
        denominator = []
        for k in range(len(kLabels)): #KLabels and HLabels assumed same dimension
            if kLabels[k] == i and hLabels[k] == j:
                numerator.append(k)
                denominator.append(k)
            elif kLabels[k] == i:
                denominator.append(k)
            elif hLabels[k] == j:
                denominator.append(k)

        print('Jaccard similarity of kmeans cluster ' + str(i) + ' hierarchical cluster ' + str(j) + ' similarity ' + str(len(numerator)/len(denominator)))

        
