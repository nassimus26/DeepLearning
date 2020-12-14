import numpy as np
from sklearn.svm import SVC

import dataset as ds

'''
 * @author Nassim MOUALEK
 * @since 06/12/2020
'''

rows = 200
features, targets = ds.getSprialDataset(rows)

print(features)
svm = SVC()
svm.fit(features, targets.ravel())

features2, targets2 = ds.getSprialDataset(rows)


print("SVM Accuracy ", svm.score(features2, targets2.ravel()))
toPredict = [[-2, -2]]
predicted = svm.predict(toPredict)
print("X=%s, Predicted=Patient is %s" % (toPredict, np.where(predicted[0], "healthy", "sick") ))
ds.show(features, targets)
