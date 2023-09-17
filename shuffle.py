import sys, pickle, numpy as np

fn = sys.argv[1]

l = pickle.load(open(fn, 'rb'))

np.random.shuffle(l)

pickle.dump(l, open(fn,'wb'))

print('shuffle finished.')