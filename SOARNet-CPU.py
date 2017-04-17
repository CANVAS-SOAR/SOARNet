import tensorflow as tf
import sys
import os
sys.path.append('../ImageLoader')
from ImageLoader import *

path = os.getcwd()
setPath("../CityScape/")  # set cwd to the CityScape data directory
xPath = "./leftImg8bit_trainvaltest/leftImg8bit/"  # path to RGB images
yPath = "./gtFine_trainvaltest/gtFine/"         # path to labeled ground truth images
# create image object and assign paths for testing and training
images = ImageLoader(trainXPattern=xPath+"train/*/*",trainYPattern=yPath+"train/*/*labelIds.png",testXPattern=xPath+"test/*/*",testYPattern=yPath+"test/*/*labelIds.png")

images.loadTrainData()	# load the file names of training data
images.loadTestData()	# load the file names of testing data

x = tf.placeholder(tf.float32, shape=(None, 1024, 2048, 3))	# x will be RGB images
y_ = tf.placeholder(tf.float32, [None, 1024, 2048])
keep = .05

def weights(shape):
	initial = tf.random_normal(shape)
	return tf.Variable(initial)

def biases(shape):
	initial = tf.random_normal(shape)
	return tf.Variable(initial)

def convolution(x, W, b, s=1):
	x = tf.nn.conv2d(x, W, strides=[1, s, s, 1], padding='SAME')
	x = tf.nn.bias_add(x, b)
	return tf.nn.relu(x)

def maxpool(x, k=2):
	return tf.nn.max_pool(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def upsample(x, k=2):
# do something...
	return x

def network(x, weights, biases, dropout):
	#x_im = tf.reshape(x, [-1, 2048, 1024, 3])

	# 3x3 conv, 3 inputs RGB, 64 outputs
	c1_1 = convolution(x, weights([3, 3, 3, 64]), biases([64]))
	# 3x3 conv, 64 inputs from last layer, 64 ouputs
	c1_2 = convolution(c1_1, weights([3, 3, 64, 64]), biases([64]))
	# reduce image size from 2048x1024 to 1024x512
	p1 = maxpool(c1_2)

	c2_1 = convolution(p1, weights([3, 3, 64, 128]), biases([128]))
	c2_2 = convolution(c2_1, weights([3, 3, 128, 128]), biases([128]))
	p2 = maxpool(c2_2)

	c3_1 = convolution(p2, weights([3, 3, 128, 256]), biases([256]))
	c3_2 = convolution(c3_1, weights([3, 3, 256, 256]), biases([256]))
	c3_3 = convolution(c3_2, weights([3, 3, 256, 256]), biases([256]))
	p3 = maxpool(c3_3)

	c4_1 = convolution(p3, weights([3, 3, 256, 512]), biases([512]))
	c4_2 = convolution(c4_1, weights([3, 3, 512, 512]), biases([512]))
	c4_3 = convolution(c4_2, weights([3, 3, 512, 512]), biases([512]))
	p4 = maxpool(c4_3)

	c5_1 = convolution(p4, weights([3, 3, 512, 512]), biases([512]))
	c5_2 = convolution(c5_1, weights([3, 3, 512, 512]), biases([512]))
	c5_3 = convolution(c5_2, weights([3, 3, 512, 512]), biases([512]))
	p5 = maxpool(c5_3)
	# end of encoding layers

	# beginning of decoding layers
	u5 = upsample(p5)
	c5_3_D = convolution(u5, weights([3, 3, 512, 512]), biases([512]))
	c5_2_D = convolution(c5_3_D, weights([3, 3, 512, 512]), biases([512]))
	c5_1_D = convolution(c5_2_D, weights([3, 3, 512, 512]), biases([512]))
   
	u4 = upsample(c5_1_D)
	c4_3_D = convolution(u4, weights([3, 3, 512, 512]), biases([512]))
	c4_2_D = convolution(c4_3_D, weights([3, 3, 512, 512]), biases([512]))
	c4_1_D = convolution(c4_2_D, weights([3, 3, 512, 512]), biases([512]))

	u3 = upsample(c4_1_D)
	c3_3_D = convolution(u3, weights([3, 3, 512, 256]), biases([256]))
	c3_2_D = convolution(c3_3_D, weights([3, 3, 256, 256]), biases([256]))
	c3_1_D = convolution(c3_2_D, weights([3, 3, 256, 256]), biases([256]))

	u2 = upsample(c3_1_D)
	c2_2_D = convolution(u2, weights([3, 3, 256, 128]), biases([128]))
	c2_1_D = convolution(c2_2_D, weights([3, 3, 128, 128]), biases([128]))

	u1 = upsample(c2_1_D)
	c1_2_D = convolution(u1, weights([3, 3, 128, 64]), biases([64]))
	c1_1_D = convolution(c1_2_D, weights([3, 3, 64, 3]), biases([3]))

	s1 = tf.nn.softmax(c1_1_D)

	return s1


# Make prediction from above model
prediction = network(x, weights, biases, keep)

# Loss and optimization
# cost = something, must be completed
# gradDescent = must be completed, minimize cost function

# Evaluation
# must be finished, determine how much was correct
# correct = tf.equal(tf.argmax(prediction,2), tf.argmax(oneHot(y_)))
# accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))  # should be okay

init = tf.global_variables_initializer()

# must be completed, actually run the tensorflow session
# with tf.Session() as sess:

