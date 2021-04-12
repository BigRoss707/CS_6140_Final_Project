import numpy as np
from sklearn.tree import DecisionTreeClassifier

# This is the raw DEA score

raw_score = np.loadtxt('raw_dea.csv',delimiter=',')



# convert raw_score to actual score
new_score = []
for i in range(3108):
    new = 1 - raw_score[i]
    new_score.append(new)

# Raw data

raw_data = np.loadtxt('raw_data.csv', delimiter=',')
cluster_label_data = np.loadtxt('combined_data_with_cluster_labels.csv',delimiter=',')


#if a score is .9 or above the county will receive a 1. The county will receive a 0 otherwise

# list of 0 and 1 assignments per county

binary_score = []

for i in range(3108):
    score = new_score[i]
    if score >= 0.8:
        binary_score.append(1)
    else:
        binary_score.append(0)

# Here we will determine the number of counties with a passing score.
count = 0

for i in range(3108):
    if binary_score[i] == 1:
        count += 1

print("Then number of passing counties is " + str(count) +".")



# Write code to split data into bins.


# first we isolate the column that gives the bin labels for k-means with 4 clusters

k_4_labels = cluster_label_data[:,18]

# Now we isolate the last column which gives the labels for k-means with 3 clusters

k_3_labels = cluster_label_data[:,19]

# now we write some code to actually split up the data.
# we will split up the dea score data as well according to the bins


# This is how we will split the raw data according to the bins.
k_4_bin_0 = []
k_4_bin_1 = []
k_4_bin_2 = []
k_4_bin_3 = []


k_3_bin_0 = []
k_3_bin_1 = []
k_3_bin_2 = []

#This is how we will split up the dea data according to the bins
k_4_bin_0_dea = []
k_4_bin_1_dea = []
k_4_bin_2_dea = []
k_4_bin_3_dea = []


k_3_bin_0_dea = []
k_3_bin_1_dea = []
k_3_bin_2_dea = []

#This is how we will split up the binary dea score according to the bins

k_4_bin_0_dea_b = []
k_4_bin_1_dea_b = []
k_4_bin_2_dea_b = []
k_4_bin_3_dea_b = []


k_3_bin_0_dea_b = []
k_3_bin_1_dea_b = []
k_3_bin_2_dea_b = []

for i in range(3108):
    x = k_4_labels[i]

    if x == 0:
        k_4_bin_0.append(raw_data[i])
        k_4_bin_0_dea.append(new_score[i])
        k_4_bin_0_dea_b.append(binary_score[i])
    if x == 1:
        k_4_bin_1.append(raw_data[i])
        k_4_bin_1_dea.append(new_score[i])
        k_4_bin_1_dea_b.append(binary_score[i])
    if x == 2:
        k_4_bin_2.append(raw_data[i])
        k_4_bin_2_dea.append(new_score[i])
        k_4_bin_2_dea_b.append(binary_score[i])
    if x ==3:
        k_4_bin_3.append(raw_data[i])
        k_4_bin_3_dea.append(new_score[i])
        k_4_bin_3_dea_b.append(binary_score[i])

for i in range(3108):
    x = k_3_labels[i]

    if x == 0:
        k_3_bin_0.append(raw_data[i])
        k_3_bin_0_dea.append(new_score[i])
        k_3_bin_0_dea_b.append(binary_score[i])
    if x == 1:
        k_3_bin_1.append(raw_data[i])
        k_3_bin_1_dea.append(new_score[i])
        k_3_bin_1_dea_b.append(binary_score[i])
    if x ==2:
        k_3_bin_2.append(raw_data[i])
        k_3_bin_2_dea.append(new_score[i])
        k_3_bin_2_dea_b.append(binary_score[i])

k_4_bin_0 = np.array(k_4_bin_0)
k_4_bin_1 = np.array(k_4_bin_1)
k_4_bin_2 = np.array(k_4_bin_2)
k_4_bin_3 = np.array(k_4_bin_3)

k_3_bin_0 = np.array(k_3_bin_0)
k_3_bin_1 = np.array(k_3_bin_1)
k_3_bin_2 = np.array(k_3_bin_2)


#for each bin in each type of clustering lets compute the average DEA scores

average_k4_b0 = sum(k_4_bin_0_dea)/len(k_4_bin_0_dea)
average_k4_b1 = sum(k_4_bin_1_dea)/len(k_4_bin_1_dea)
average_k4_b2 = sum(k_4_bin_2_dea)/len(k_4_bin_2_dea)
average_k4_b3 = sum(k_4_bin_3_dea)/len(k_4_bin_3_dea)

average_k3_b0 = sum(k_3_bin_0_dea)/len(k_3_bin_0_dea)
average_k3_b1 = sum(k_3_bin_1_dea)/len(k_3_bin_1_dea)
average_k3_b2 = sum(k_3_bin_2_dea)/len(k_3_bin_2_dea)

print("Average Dea k_4_b_0: ",average_k4_b0)
print("Average Dea k_4_b_1: ",average_k4_b1)
print("Average Dea k_4_b_2: ",average_k4_b2)
print("Average Dea k_4_b_3: ",average_k4_b3)

print("Average Dea k_3_b_0: ",average_k3_b0)
print("Average Dea k_3_b_1: ",average_k3_b1)
print("Average Dea k_3_b_2: ",average_k3_b2)

# for each bin let's count the number of scores that are considered passing

pass_k_4_b_0 = 0
pass_k_4_b_1 = 0
pass_k_4_b_2 = 0
pass_k_4_b_3 = 0

pass_k_3_b_0 = 0
pass_k_3_b_1 = 0
pass_k_3_b_2 = 0

for i in range(len(k_4_bin_0_dea_b)):
    if k_4_bin_0_dea_b[i] == 1:
        pass_k_4_b_0 += 1

for i in range(len(k_4_bin_1_dea_b)):
    if k_4_bin_1_dea_b[i] == 1:
        pass_k_4_b_1 += 1

for i in range(len(k_4_bin_2_dea_b)):
    if k_4_bin_2_dea_b[i] == 1:
        pass_k_4_b_2 += 1

for i in range(len(k_4_bin_3_dea_b)):
    if k_4_bin_3_dea_b[i] == 1:
        pass_k_4_b_3 += 1

for i in range(len(k_3_bin_0_dea_b)):
    if k_3_bin_0_dea_b[i] == 1:
        pass_k_3_b_0 += 1

for i in range(len(k_3_bin_1_dea_b)):
    if k_3_bin_1_dea_b[i] == 1:
        pass_k_3_b_1 += 1

for i in range(len(k_3_bin_2_dea_b)):
    if k_3_bin_2_dea_b[i] == 1:
        pass_k_3_b_2 += 1

print("Number passing in k_4_b_0: ",pass_k_4_b_0)
print("Number passing in k_4_b_1: ",pass_k_4_b_1)
print("Number passing in k_4_b_2: ",pass_k_4_b_2)
print("Number passing in k_4_b_3: ",pass_k_4_b_3)

print("Number passing in k_3_b_0: ",pass_k_3_b_0)
print("Number passing in k_3_b_1: ",pass_k_3_b_1)
print("Number passing in k_3_b_2: ",pass_k_3_b_2)

# Let's figure out the percent that are passing in each cluster
print("percent passing in k_4_b_0: ",pass_k_4_b_0/len(k_4_bin_0))
print("percent passing in k_4_b_1: ",pass_k_4_b_1/len(k_4_bin_1))
print("percent passing in k_4_b_2: ",pass_k_4_b_2/len(k_4_bin_2))
print("percent passing in k_4_b_3: ",pass_k_4_b_3/len(k_4_bin_3))

print("percent passing in k_3_b_0: ",pass_k_3_b_0/len(k_3_bin_0))
print("percent passing in k_3_b_1: ",pass_k_3_b_1/len(k_3_bin_1))
print("percent passing in k_3_b_2: ",pass_k_3_b_2/len(k_3_bin_2))

#now we want to see which attribute had the biggest affect on a county receiving a passing score in a particular cluster.
#we will use a decision tree analysis to do this.

tree = DecisionTreeClassifier(criterion = "entropy", max_depth = 3)
k_4_b_0_tree = tree.fit(k_4_bin_0,k_4_bin_0_dea_b).feature_importances_
k_4_b_1_tree = tree.fit(k_4_bin_1,k_4_bin_1_dea_b).feature_importances_
k_4_b_2_tree = tree.fit(k_4_bin_2,k_4_bin_2_dea_b).feature_importances_
k_4_b_3_tree = tree.fit(k_4_bin_3,k_4_bin_3_dea_b).feature_importances_

k_3_b_0_tree = tree.fit(k_3_bin_0,k_3_bin_0_dea_b).feature_importances_
k_3_b_1_tree = tree.fit(k_3_bin_1,k_3_bin_1_dea_b).feature_importances_
k_3_b_2_tree = tree.fit(k_3_bin_2,k_3_bin_2_dea_b).feature_importances_
print("k=4 clustering")
print(k_4_b_0_tree)
print(k_4_b_1_tree)
print(k_4_b_2_tree)
print(k_4_b_3_tree)
print("k=3 clustering")
print(k_3_b_0_tree)
print(k_3_b_1_tree)
print(k_3_b_2_tree)


print("the level of importance of attributes for k_4_b_0:",16,15,10,8)
print("the level of importance of attributes for k_4_b_1:",15,16,0,4)
print("the level of importance of attributes for k_4_b_2:",16,15,0)
print("the level of importance of attributes for k_4_b_3:",15,16,2,8)

print("the level of importance of attributes for k_3_b_0:",15,16)
print("the level of importance of attributes for k_3_b_1:",15,16,10,0)
print("the level of importance of attributes for k_3_b_2:",16,15)