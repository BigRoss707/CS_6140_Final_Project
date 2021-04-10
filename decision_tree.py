from sklearn.tree import DecisionTreeClassifier
import numpy as np
import clusters

num_clusters = 4
data = clusters.get_data()

tree = DecisionTreeClassifier(criterion="entropy", max_depth=10)
tree.fit(data, clusters.labels(num_clusters))
