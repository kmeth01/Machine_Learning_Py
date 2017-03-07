import sys
from sklearn import preprocessing
import numpy as np
from PIL import Image

print 'Number of arguments:', len(sys.argv), 'arguments.'
print 'Argument List:', str(sys.argv)
print sys.argv[1]
#Reading Input Arguments
K=int(sys.argv[1])
InputImage=Image.open(sys.argv[2])
InputImage.load()
OutputFilename=sys.argv[3]
data=np.asarray(InputImage,dtype="int32")
rgb_im=InputImage.convert('RGB')
width,height=InputImage.size
print width
print height
print K
data_arr=[]
for i in range(data.shape[0]):
    for j in range(data.shape[1]):
        data_arr.append((np.c_[np.array([data[i,j,:]]),np.array([[i,j]])]).reshape(5,1))
data_arr=np.asarray(data_arr)
r,c,ex=data_arr.shape

min_max_scaler=preprocessing.MinMaxScaler()
data_arr_std=min_max_scaler.fit_transform(data_arr.reshape(r,c))
#for i in range(r):
#    print data_arr_std[i,:]
np.random.seed(42)
idx=np.arange(r)
np.random.shuffle(idx)
X=data_arr_std[idx]
centroids=X[:K]
oldcentroids=np.zeros((K,5))
ledger_new=np.zeros(r,dtype=np.int)
ledger_old=np.ones(r)
while(np.all(oldcentroids!=centroids)):
    oldcentroids=centroids
    for i in range(r):
        dist=np.zeros(K)
        for j in range(K):
            dist[j]=np.linalg.norm(data_arr_std[i]-centroids[j])
        ledger_new[i]=int(np.argmin(dist))
    centroids=np.zeros((K,5))
    for i in range(r):
        centroids[ledger_new[i]]=centroids[ledger_new[i]]+data_arr_std[i]
    for j in range(K):
        n=np.sum(ledger_new==j)
        centroids[j]=centroids[j]/float(n)
for i in range(r):
    data_arr_std[i][:3]=centroids[ledger_new[i]][:3]
for i in range(K):
    print centroids[i]
#img=Image.fromarray(data,'RGB')
#img.show()
for i in range(data_arr_std.shape[0]):
      m,n=data_arr[i][3:5]
      #print m,n
      #print data[m,n,:]
      data[m,n,:]=centroids[ledger_new[i]][:3]*255
img=Image.fromarray(data.astype(np.uint8),'RGB')
img.save(OutputFilename)