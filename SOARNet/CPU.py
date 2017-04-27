import tensorflow as tf
import sys
import os


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
    return tf.nn.max_pool_with_argmax(x, ksize=[1,k,k,1], strides=[1,k,k,1], padding='SAME')

def upsample(x, indices, k=2):
    # do something...
    return x

def SOARNet(x, dropout, num_classes):

    # 3x3 conv, 3 inputs RGB, 64 outputs
    c1_1 = convolution(x, weights([3, 3, 3, 64]), biases([64]))
    # 3x3 conv, 64 inputs from last layer, 64 ouputs
    c1_2 = convolution(c1_1, weights([3, 3, 64, 64]), biases([64]))
    # reduce image size from 2048x1024 to 1024x512
    p1, i1 = maxpool(c1_2)

    c2_1 = convolution(p1, weights([3, 3, 64, 128]), biases([128]))
    c2_2 = convolution(c2_1, weights([3, 3, 128, 128]), biases([128]))
    p2, i2 = maxpool(c2_2)

    c3_1 = convolution(p2, weights([3, 3, 128, 256]), biases([256]))
    c3_2 = convolution(c3_1, weights([3, 3, 256, 256]), biases([256]))
    c3_3 = convolution(c3_2, weights([3, 3, 256, 256]), biases([256]))
    p3, i3 = maxpool(c3_3)

    c4_1 = convolution(p3, weights([3, 3, 256, 512]), biases([512]))
    c4_2 = convolution(c4_1, weights([3, 3, 512, 512]), biases([512]))
    c4_3 = convolution(c4_2, weights([3, 3, 512, 512]), biases([512]))
    p4, i4 = maxpool(c4_3)

    c5_1 = convolution(p4, weights([3, 3, 512, 512]), biases([512]))
    c5_2 = convolution(c5_1, weights([3, 3, 512, 512]), biases([512]))
    c5_3 = convolution(c5_2, weights([3, 3, 512, 512]), biases([512]))
    p5, i5 = maxpool(c5_3)
    # end of encoding layers

    # beginning of decoding layers
    u5 = upsample(p5, i5)
    c5_3_D = convolution(u5, weights([3, 3, 512, 512]), biases([512]))
    c5_2_D = convolution(c5_3_D, weights([3, 3, 512, 512]), biases([512]))
    c5_1_D = convolution(c5_2_D, weights([3, 3, 512, 512]), biases([512]))

    u4 = upsample(c5_1_D, i4)
    c4_3_D = convolution(u4, weights([3, 3, 512, 512]), biases([512]))
    c4_2_D = convolution(c4_3_D, weights([3, 3, 512, 512]), biases([512]))
    c4_1_D = convolution(c4_2_D, weights([3, 3, 512, 256]), biases([256]))

    u3 = upsample(c4_1_D, i3)
    c3_3_D = convolution(u3, weights([3, 3, 256, 256]), biases([256]))
    c3_2_D = convolution(c3_3_D, weights([3, 3, 256, 256]), biases([256]))
    c3_1_D = convolution(c3_2_D, weights([3, 3, 256, 128]), biases([128]))

    u2 = upsample(c3_1_D, i2)
    c2_2_D = convolution(u2, weights([3, 3, 128, 128]), biases([128]))
    c2_1_D = convolution(c2_2_D, weights([3, 3, 128, 64]), biases([64]))

    u1 = upsample(c2_1_D, i1)
    c1_2_D = convolution(u1, weights([3, 3, 64, 64]), biases([64]))
    c1_1_D = convolution(c1_2_D, weights([3, 3, 64, num_classes]), biases([num_classes]))

    # To get prediction, use tf.nn.softmax()
    # To get error, use tf.nn.softmax_cross_entropy_with_logits
    return c1_1_D


