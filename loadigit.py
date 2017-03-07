import numpy as np
#from skdata.mnist.views import OfficialImageClassification
from sklearn.manifold import TSNE
from matplotlib import pyplot as plt
from t_sne import bh_sne

# load up data
# load the data set
# load the data
filename = 'C:\Users\karth\Desktop\CIS_519\hw3_skeleton\hw3_skeleton\data\challengeTrainLabeled.dat'
filename_T='C:\Users\karth\Desktop\CIS_519\hw3_skeleton\hw3_skeleton\data\challengeTestUnlabeled.dat'
allData = np.loadtxt(filename, delimiter=',')
allData_T = np.loadtxt(filename_T, delimiter=',')

x_data = allData[:,:-1]
y_data = allData[:,-1]


# convert image data to float64 matrix. float64 is need for bh_sne
x_data = np.asarray(x_data).astype('float64')
x_data = x_data.reshape((x_data.shape[0], -1))


# perform t-SNE embedding
vis_data = bh_sne(x_data)

# plot the result
vis_x = vis_data[:, 0]
vis_y = vis_data[:, 1]

plt.scatter(vis_x, vis_y, c=y_data, cmap=plt.cm.get_cmap("jet", 10))
plt.colorbar(ticks=range(10))
plt.clim(-0.5, 9.5)
plt.show()