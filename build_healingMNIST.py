from sklearn.datasets import fetch_mldata
import numpy as np
import matplotlib.pyplot as plt
import cv2
import os

mnist = fetch_mldata('MNIST original', data_home="data")

# 
num_data=100
#p = np.random.random_integers(0, len(mnist.data), 1)
idx=list(range(len(mnist.data)))
np.random.shuffle(idx)
p = idx[0:num_data]
#p = list(range(25))
mnist_arr=np.array(list(zip(mnist.data, mnist.target)))
n_steps=20
healing_data=[]
healing_label=[]
healing_steps=[]
for index, val in enumerate(mnist_arr[p]):
	data=val[0]
	label=val[1]
	img=data.reshape(28,28)
	# 画像サイズの取得(横, 縦)
	size = tuple([img.shape[1], img.shape[0]])
	center = tuple([int(size[0]/2), int(size[1]/2)])
	scale = 1.0
	seq=[]
	for j in range(n_steps):
		angle =10.0*j
		# 回転変換
		rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
		img_rot = cv2.warpAffine(img, rotation_matrix, size, flags=cv2.INTER_CUBIC)
		seq.append(img_rot.reshape(28*28))
	healing_steps.append(n_steps)
	healing_label.append(label)
	healing_data.append(seq)
x=np.array(healing_data,dtype=np.float32)
y=np.array(healing_label,dtype=np.int32)
steps=np.array(healing_steps,dtype=np.int32)
print(x.shape)
print(y.shape)
save_path="healingMNIST"
os.makedirs(save_path,exist_ok=True)

mask=np.ones(x.shape[0:2],dtype=np.float32)

save_file=save_path+"/data.npy"
np.save(save_file,x)
save_file=save_path+"/label.npy"
np.save(save_file,y)
save_file=save_path+"/mask.npy"
np.save(save_file,mask)
save_file=save_path+"/steps.npy"
np.save(save_file,steps)


#
#plt.subplot(5, 5, index + 1)
#plt.subplot(5, 5, j + 1)
#plt.axis('off')
#plt.imshow(data.reshape(28, 28), cmap=plt.cm.gray_r, interpolation='nearest')
#plt.title('%i' % label)
#plt.show()

