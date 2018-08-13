import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.preprocessing import image
from keras.models import Sequential
from keras.layers import Conv2D, Lambda
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout
from keras.utils.np_utils import to_categorical

dataset = pd.read_csv('train.csv')
X_test = pd.read_csv('test.csv')

images = (dataset.iloc[:10,1:].values).astype('float32')
labels = (dataset.iloc[:10,:1].values).astype('int32')
X_test = X_test.values.astype('float32')

images = images.reshape((images.shape[0],28,28,1))
mean_px = images.mean().astype(np.float32)
std_px = images.std().astype(np.float32)

labels = to_categorical(labels)

# train_images,test_images,train_labels,test_labels = train_test_split(images,labels,random_state=0)

gen = image.ImageDataGenerator()

batches = gen.flow(images, labels, batch_size=64)
# val_batches = gen.flow(test_images, test_labels, batch_size=64)

def standardize(x): 
    return (x-mean_px)/std_px

def CNN_Model():
	clf = Sequential()
	clf.add(Lambda(standardize, input_shape=(28,28,1)))
	clf.add(Conv2D(512,(3,3), activation='relu'))
	# clf.add(Conv2D(64,(3,3), activation='relu'))

	clf.add(MaxPooling2D(pool_size=(2,2)))
	clf.add(Conv2D(256,(3,3), activation='relu'))
	# clf.add(Conv2D(128,(3,3), activation='relu'))

	# clf.add(Dropout(0.2))
	clf.add(MaxPooling2D(pool_size=(2,2)))

	clf.add(Conv2D(128,(3,3), activation='relu'))


	# clf.add(Dropout(0.2))
	clf.add(MaxPooling2D(pool_size=(2,2)))
	clf.add(Flatten())
	# clf.add(Dropout(0.2))
	clf.add(Dense(64, activation='relu'))
	clf.add(Dense(10, activation='softmax'))

	clf.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
	clf.fit_generator(generator=batches, steps_per_epoch=batches.n, epochs=1)
	return clf

model = CNN_Model()

X_test = X_test.reshape(X_test.shape[0], 28, 28,1)

predictions = model.predict_classes(X_test, verbose=1)

model.save('digit_recognizer_model.h5')

submissions=pd.DataFrame({"ImageId": list(range(1,len(predictions)+1)),
                         "Label": predictions})
submissions.to_csv("DR.csv", index=False, header=True)

