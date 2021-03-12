import numpy as np

load_scores = np.loadtxt('dea_scores.csv')

new_scores = []
for score in load_scores:
    new_score = 1-score
    if new_score <0:
        new_score = 0
    new_scores.append(new_score)

new_scores = np.array(new_scores)
new_scores = new_scores.transpose()

np.savetxt('final_dea_score.csv',new_scores)