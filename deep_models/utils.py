from tensorflow.keras.datasets import cifar10
import numpy as np 
import cv2 as cv 
from tensorflow.keras.utils import to_categorical 
from tensorflow.keras.layers import Conv2D,MaxPool2D,concatenate
from tensorflow.keras.initializers import glorot_uniform,Constant
from tensorflow.nn import relu

kernel_init,bias_init = glorot_uniform(),Constant(value = 0.2)

def load_cifar10_data(img_rows,img_cols):
	(X_train,y_train),(X_test,y_test) = cifar10.load_data()

	X_train	= np.array([cv.resize(img, (img_rows,img_cols)) for img in X_train[:,:,:,:]])
	X_test	= np.array([cv.resize(img, (img_rows,img_cols)) for img in X_test[:,:,:,:]])

	X_train = np.array([cv.resize(img,(img_rows,img_cols)) for img in X_train[:,:,:,:]])
	X_test = np.array([cv.resize(img,(img_rows,img_cols)) for img in X_test[:,:,:,:]])


	y_train = to_categorical(y_train,10)
	y_test = to_categorical(y_test,10)
	X_train = X_train / 255.0
	X_test = X_test / 255.0 

	return (X_train,y_train),(X_test,y_test)


def inception_module(x,
					filters_1x1,
					filters_3x3_a,
					filters_3x3_b,
					filters_5x5_a,
					filters_5x5_b,
					filters_pool,
					name=None):
	conv_1x1 = Conv2D(filters = filters_1x1,kernel_size = (1,1),padding = 'same', activation = relu,kernel_initializer = kernel_init,bias_initializer = bias_init)(x)

	conv_3x3 = Conv2D(filters = filters_3x3_a,kernel_size = (1,1),padding = 'same', activation = relu,kernel_initializer = kernel_init,bias_initializer = bias_init)(x)

	conv_3x3 = Conv2D(filters = filters_3x3_b,kernel_size = (3,3),padding = 'same', activation = relu,kernel_initializer = kernel_init,bias_initializer = bias_init)(conv_3x3)

	conv_5x5 = Conv2D(filters = filters_5x5_a,kernel_size = (1,1),padding = 'same', activation = relu,kernel_initializer = kernel_init,bias_initializer = bias_init)(x)

	conv_5x5 = Conv2D(filters = filters_5x5_b,kernel_size = (5,5),padding = 'same', activation = relu,kernel_initializer = kernel_init,bias_initializer = bias_init)(conv_5x5)	

	pool_ = MaxPool2D((3,3), strides=(1,1),padding = 'same')(x)

	pool_ = Conv2D(filters = filters_pool,kernel_size = (1,1),padding = 'same', activation = relu,kernel_initializer = kernel_init,bias_initializer = bias_init)(pool_)	

	return concatenate([conv_1x1,conv_3x3,conv_5x5,pool_],axis = 3,name = name)
