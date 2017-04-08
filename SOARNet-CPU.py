import tensorflow as tf
import sys
sys.path.append('../ImageLoader')
from ImageLoader import ImageLoader as il

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
	return

def network(x, weights, biases, dropout):
	c1_1 = convolution(x, weights([3, 3, 2048, 1024]), biases([64]))
	c1_2 = convolution(c1_1, weights([3, 3, 2048, 1024]), biases([64]))
	p1 = maxpool(c1_2)

	c2_1 = convolution(p1, weight_variable([3, 3, 1024, 512]), biases([128]))
	c2_2 = convolution(c2_1, weight_variable([3, 3, 1024, 512]), biases([128]))
	p2 = maxpool(c2_2)

	c3_1 = convolution(p2, weights([3, 3, 512, 256]), biases([256]))
	c3_2 = convolution(c3_1, weights([3, 3, 512, 256]), biases([256]))
	c3_3 = convolution(c3_2, weights([3, 3, 512, 256]), biases([256]))
	p3 = maxpool(c3_3)

	c4_1 = convolution(p3, weights([3, 3, 256, 128]), biases([512]))
	c4_2 = convolution(c4_1, weights([3, 3, 256, 128]), biases([512]))
	c4_3 = convolution(c4_2, weights([3, 3, 256, 128]), biases([512]))
	p4 = maxpool(c4_3)

	c5_1 = convolution(p4, weights([3, 3, 128, 64]), biases([512]))
	c5_2 = convolution(c5_1, weights([3, 3, 128, 64]), biases([512]))
	c5_3 = convolution(c5_2, weights([3, 3, 128, 64]), biases([512]))
	p5 = maxpool(c5_3)

	u1 = upsample(p5)
	c6_1 = convolution(u1, weights([3, 3, 128, 64]), biases([512]))
	c6_2 = convolution(c6_1, weights([3, 3, 128, 64]), biases([512]))
	c6_3 = convolution(c6_2, weights([3, 3, 128, 64]), biases([512]))
   
	u2 = upsample(c6_3)
	c7_1 = convolution(u2, weights([3, 3, 256, 128]), biases([512]))
	c7_2 = convolution(c7_1, weights([3, 3, 256, 128]), biases([512]))
	c7_3 = convolution(c7_2, weights([3, 3, 256, 128]), biases([512]))

	u3 = upsample(c7_3)
	c8_1 = convolution(u3, weights([3, 3, 512, 256]), biases([256]))
	c8_2 = convolution(c8_1, weights([3, 3, 512, 256]), biases([256]))
	c8_3 = convolution(c8_2, weights([3, 3, 512, 256]), biases([256]))

	u4 = upsample(c8_3)
	c9_1 = convolution(u4, weights([3, 3, 1024, 512]), biases([128]))
	c9_2 = convolution(c9_1, weights([3, 3, 1024, 512]), biases([64]))

	u5 = upsample(c9_2)
	c10_1 = convolution(u5, weights([3, 3, 2048, 1024]), biases([64]))
	c10_2 = convolution(c10_1, weights([3, 3, 2048, 1024]), biases([3]))

	s1 = tf.nn.softmax(p5)

	return s1

