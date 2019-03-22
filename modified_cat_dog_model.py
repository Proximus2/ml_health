import numpy as np 
import matplotlib.pyplot as plt 
import os
import cv2


def prepare_data():
	image_size = 50
	training_data = []
	CATEGORIES =["Dog","Cat"]

	# CATEGORIES =["positive","negative"]
	for category in CATEGORIES:
		# getting the folder that contains a specific category of images e.g the dog folder or cat folder
		path = os.path.join(os.getcwd(),"PetImages/"+category)
		# taking index of the category as the class name
		# this simplifies the algorithm since index is numeric
		class_name = CATEGORIES.index(category)
		
		for img in os.listdir(path):
			# removing the color by adding the grayscale
			try:
				img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
				# to see the transformed image you can remove the comments on the code below

				# plt.imshow(img_array,cmap="gray")
				# plt.show()

				# here we resize the image so as to ensure same sized images are fed into the network (easier to handle than allowing for different sized images)
				new_img_array = cv2.resize(img_array,(image_size,image_size))

				# we store the new image ready for training
				training_data.append([new_img_array,class_name])
				# plt.imshow(new_imgarray,cmap="gray")
				# plt.show()
				# print(img_array)

			except Exception as e:
				pass

	#this shows simply how many images you have successfully pre-process without any errors			
	print(len(training_data))

	import random

	# we randomize the training data to reduce model bias ...
	random.shuffle(training_data)

	# preparing the set of features(independent) and labels(dependent) variables
	# this is doen by splitting the newly acquired training data set
	x_train=[]
	y_train=[]

	# separate the features and labels from the trainind data
	for features,label in training_data:
		x_train.append(features)
		y_train.append(label)

				
	# converte the features into a numpy array
	# reshape(number of features (-1) means any number, size,size,(1 for grey scale 3 for color images))
	x_train = np.array(x_train).reshape(-1,image_size,image_size,1)
		

	import pickle
	# storing the newly prepared features and labels as binary file for easier future reference


	# pickle_save = open("cervical_cancer_features.pickle","wb")

	pickle_save = open("cat_dog_features.pickle","wb")
	pickle.dump(x_train,pickle_save)
	pickle_save.close()

	# pickle_save = open("cervical_cancer_labels.pickle","wb")
	pickle_save = open("cat_dog_labels.pickle","wb")

	pickle.dump(y_train,pickle_save)
	pickle_save.close()



def prepare_train_model(model_name="model_nameX1"):

	import tensorflow as tf 
	from tensorflow.keras.models import Sequential
	from tensorflow.keras.layers import Dense,Activation,Dropout,Flatten,Conv2D,MaxPooling2D
	import pickle
	import time

	# reading in the binary files containing the already prepared features (images) and labels

	x_train = pickle.load(open("cat_dog_features.pickle","rb"))
	y_train = pickle.load(open("cat_dog_labels.pickle","rb"))

	# normalize the data , i.e putting it in range of o to 1, since this is imagery data , pixel range from 0 to 255, thus we can just divide the data set by 255 to get the normalized data

	x_train = x_train/255.0

	# the begining of the neural network

	model = Sequential()

	# layer 1
	# input layer
	model.add(Conv2D(64,(3,3),input_shape=x_train.shape[1:]))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))

	# layer2
	model.add(Conv2D(64,(3,3)))
	model.add(Activation("relu"))
	model.add(MaxPooling2D(pool_size=(2,2)))

	# layer3
	model.add(Flatten())
	model.add(Dense(64))
	model.add(Activation("relu"))

	# output layer
	model.add(Dense(1))
	model.add(Activation("sigmoid"))


	model.compile(loss="binary_crossentropy",optimizer="adam",metrics=["accuracy"])

	# the end of the neural network

	# model.fit(x_train,y_train,epochs=10,batch_size=40,validation_split=0.1,callbacks=[tensorboard])

	# the actual training 
	model.fit(x_train,y_train,epochs=10,batch_size=40,validation_split=0.1)


	# model.save("64x3-CNN-MODEL")

	# once the training is done we save the model
	# for later reference and usage
	model.save(model_name)




def model_usage(model_name="model_nameX1",image_location="image.jpeg"):
	import cv2
	import tensorflow as tf 

	CATEGORIES =["Dog","Cat"]
	# CATEGORIES =["positive","negative"]

	def prepare(filepath):
		image_size = 50
		img_array = cv2.imread(filepath,cv2.IMREAD_GRAYSCALE)
		new_array = cv2.resize(img_array,(image_size,image_size))

		return new_array.reshape(-1,image_size,image_size,1)


	model = tf.keras.models.load_model(model_name)
	prediction = model.predict([prepare(image_location)])
	print(CATEGORIES[int(prediction[0][0])])	



def execute():
	prepare_data()
	prepare_train_model(model_name="dog_cat_model_1")


execute()

