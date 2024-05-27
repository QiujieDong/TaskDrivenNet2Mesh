import os
import shutil
import numpy as np

simply_path = '/home/qiujie/LaplacianNN/data/human_trainDense_testSimply/segs/test/shrec1'

origin_path = '/home/qiujie/LaplacianNN/data/human_trainDense_testSimply/segs/test/shurec'


for f in sorted(os.listdir(simply_path)):

    f_path = os.path.join(simply_path, f)

    labels = (np.loadtxt(f_path).astype(int) + 1)

    origin_f_path = os.path.join(origin_path, 'shrec_{}_full.txt'.format((os.path.splitext(f)[0]).split('__')[-1]))

    np.savetxt(origin_f_path, labels)

a = np.loadtxt('/home/qiujie/LaplacianNN/data/human_trainDense_testSimply/segs/test/shurec/shrec_8_full.txt').astype(int) - 1

print(1)