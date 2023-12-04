import sys
import numpy as np
from scipy.sparse import csr_matrix
from xclib.data import data_utils
from xclib.evaluation import xc_metrics

dataset = sys.argv[1]

# Read file with features and labels
train_features, train_labels, num_samples, num_features, num_labels = data_utils.read_data(f'data/{dataset}/train.txt')
test_features, test_labels, _, _, _ = data_utils.read_data(f'data/{dataset}/test.txt')

if test_features.shape[1] != train_features.shape[1]:
    test_features = csr_matrix(test_features, shape=(test_features.shape[0], train_features.shape[1]))

train_labels = train_labels.astype('int32')
test_labels = test_labels.astype('int32')

if dataset.startswith('wikipedia'):
    a, b = 0.5, 0.4
elif dataset.startswith('amazon'):
    a, b = 0.6, 2.6
else:
    a, b = 0.55, 1.5

inv_propen = xc_metrics.compute_inv_propesity(train_labels, a, b)
np.savetxt(f'data/{dataset}/inv_prop.txt', inv_propen)

data_utils.write_sparse_file(train_features, f'data/{dataset}/trn_X_Xf.txt')
data_utils.write_sparse_file(train_labels, f'data/{dataset}/trn_X_Y.txt')
data_utils.write_sparse_file(test_features, f'data/{dataset}/tst_X_Xf.txt')
data_utils.write_sparse_file(test_labels, f'data/{dataset}/tst_X_Y.txt')


### append label to train data

trn_X_file = open(f'data/{dataset}/trn_X_Xf.txt', 'r')
trn_Y_file = open(f'data/{dataset}/trn_X_Y.txt', 'r')
fout = open(f'data/{dataset}/trn_X_XY.txt', 'w')

num_trn, num_dim = map(int, trn_X_file.readline().split())
_, num_lab = map(int, trn_Y_file.readline().split())

offset = num_dim

fout.write(str(num_trn)+' '+str(num_dim+num_lab)+'\n')

lines_X = trn_X_file.readlines()
lines_Y = trn_Y_file.readlines()

for lineX, lineY in zip(lines_X, lines_Y):
	finalstr = lineX.strip()

	for each in lineY.split():
		lab = int(each.split(':')[0])
		finalstr += ' ' + str(offset + lab) + ':1'

	fout.write(finalstr+'\n')

trn_X_file.close()
trn_Y_file.close()
fout.close()