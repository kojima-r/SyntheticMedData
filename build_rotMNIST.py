from sklearn.datasets import fetch_mldata
import numpy as np
import cv2
import os
import json
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from numpy.random import *


mnist = fetch_mldata('MNIST original', data_home="data")

# 
num_data=-1
n_steps=20
base_path=""
save_path="rotMNIST/data/"
save_datafile="rotMNIST/rotMNIST.json"
save_labelfile="rotMNIST/rotMNIST.label.json"
plot_num=10
plot_steps=20
save_plt="rotMNIST/rotMNIST.png"
seed(1234)

os.makedirs(save_path,exist_ok=True)
#p = np.random.random_integers(0, len(mnist.data), 1)
idx=list(range(len(mnist.data)))
np.random.shuffle(idx)
if num_data>0:
	idx = idx[0:num_data]
#
# rotMNIST.label.json
rot_label_data={}
rot_label_mapping={}
for i in range(10):
	rot_label_data[i]=str(i)+"L"
	rot_label_data[i+10]=str(i)+"R"
	rot_label_mapping[rot_label_data[i]]=i
	rot_label_mapping[rot_label_data[i+10]]=i+10
fp=open(save_labelfile,"w")
json.dump(rot_label_data,fp,indent=4)
print("[SAVE]",save_labelfile)


#
mnist_arr=np.array(list(zip(mnist.data, mnist.target)))
rot_data={}
plot_data=[]
plot_label=[]
for index, val in enumerate(mnist_arr[idx]):
	data=val[0]
	target_label=int(val[1])
	img=data.reshape(28,28)
	# 画像サイズの取得(横, 縦)
	size = tuple([img.shape[1], img.shape[0]])
	center = tuple([int(size[0]/2), int(size[1]/2)])
	scale = 1.0
	offset_angle=rand()*360
	
	if rand()>0.5:
		# L
		velocity=10.0
		label=str(target_label)+"L"
	else:
		# R
		velocity=-10.0
		label=str(target_label)+"R"
	seq=[]
	for j in range(n_steps):
		angle =velocity*j+offset_angle
		# 回転変換
		rotation_matrix = cv2.getRotationMatrix2D(center, angle, scale)
		img_rot = cv2.warpAffine(img, rotation_matrix, size, flags=cv2.INTER_CUBIC)
		seq.append(img_rot.reshape(28*28))
	if len(plot_data)<plot_num:
		plot_data.append(seq[0:plot_steps])
		plot_label.append(label)
	
	save_file=save_path+"/"+"%06d"%(index)+".npy"
	np.save(save_file,seq)
	print("[SAVE]",save_file)
	rot_data[save_file]={}
	rot_data[save_file]["data"]=[base_path+save_file]
	rot_data[save_file]["label"]=rot_label_mapping[label]
fp=open(save_datafile,"w")
json.dump(rot_data,fp,indent=4)
print("[SAVE]",save_datafile)

cnt=1
for index in range(plot_num):
	data=plot_data[index]
	for j in range(plot_steps):
		plt.subplot(plot_num, plot_steps,cnt)
		plt.axis('off')
		plt.imshow(data[j].reshape(28, 28), cmap=plt.cm.gray, interpolation='nearest')
		cnt+=1
#plt.title('%i' % label)
#plt.legend(loc='lower right')
plt.savefig(save_plt)
print("[SAVE]",save_plt)
plt.clf()


