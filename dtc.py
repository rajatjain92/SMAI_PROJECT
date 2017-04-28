import numpy as np
import os
import data
import pickle
import decisionTree

x = []
x_r = []
y = []

for vid in os.listdir('dat'):
    f = open('dat/' + vid, 'rb')
    dat = pickle.load(f)
    f.close()

    datp = [0] * 5
    for i in range(len(dat) - 1):
        for j in range(5):
            if dat[i][j] > dat[i + 1][j]:
                datp[j] += 1

    x.append(datp)
    y.append(1 if data.get_class(vid) else -1)

    # reverse data
    datp = [0] * 5
    for i in range(len(dat) - 1):
        for j in range(5):
            if dat[i + 1][j] > dat[i][j]:
                datp[j] -= 1

    x_r.append(datp)
    
data = np.array(x).T.copy()
data_r = np.array(x_r).T.copy()
label = np.array(y)

num_iter = 100
k = 5

score = []
fp = []
fn = []
tp = []
tn = []

for iter_no in range(num_iter):
    print('Iter. no.', iter_no)
    rand_perm = np.random.permutation(data.shape[1])
    val_perm = rand_perm[:data.shape[1]//k]
    tr_perm = rand_perm[data.shape[1]//k:]
    tr_data = np.hstack((data[:, tr_perm], data_r[:, tr_perm]))
    tr_label = np.hstack((label[tr_perm], -label[tr_perm]))
    val_data = data[:, val_perm]
    val_data_r = data_r[:, val_perm]
    val_label = label[val_perm]

    # random flip data
    rand_sign = np.random.random_sample(val_label.size) < 0.5
    val_data[:, rand_sign] = val_data_r[:, rand_sign]
    val_label[rand_sign] *= -1
    
    s = decisionTree.DecisionTree(score_func=decisionTree.chi_score, max_depth = 10)
    s.train(tr_data, tr_label)
    prediction = s.predict(val_data)
    
    score.append(np.count_nonzero(prediction == val_label)/len(val_label))
    tp.append(np.count_nonzero((prediction == 1) & (val_label == 1)))
    fp.append(np.count_nonzero((prediction == 1) & (val_label == -1)))
    tn.append(np.count_nonzero((prediction == -1) & (val_label == -1)))
    fn.append(np.count_nonzero((prediction == -1) & (val_label == 1)))

print(np.mean(score))
