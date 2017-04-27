import tensorflow as tf
import os
import sys

from SOARNet.CPU import SOARNet
from ImageLoader import ImageLoader

sys.path.append('../ImageLoader')
from ImageLoader import *


def main():
    path = os.getcwd()
    setPath("../CityScape/")  # set cwd to the CityScape data directory
    xPath = "./leftImg8bit_trainvaltest/leftImg8bit/"  # path to RGB images
    yPath = "./gtFine_trainvaltest/gtFine/"         # path to labeled ground truth images

    # create image loader object and assign paths for testing and training
    images = ImageLoader(trainXPattern=xPath+"train/*/*",trainYPattern=yPath+"train/*/*labelIds.png",testXPattern=xPath+"test/*/*",testYPattern=yPath+"test/*/*labelIds.png")

    images.loadTrainData()	# load the file names of training data
    images.loadTestData()	# load the file names of testing data

    x = tf.placeholder(tf.float32, shape=(None, 1024, 2048, 3))	# x will be RGB images
    y_ = tf.placeholder(tf.float32, [None, 1024, 2048])
    keep = .05
    # Import data

    # Create the model
    y = SOARNet(x, keep, num_classes)

    # Loss function
    cross_entropy = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y))

    #Training algorithm
    train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy)

    #Initialize the graph
    sess = tf.InteractiveSession()
    tf.global_variables_initializer().run()

    # Train
    for i in range(1000):
        batch_xs, batch_ys = images.getNextBatch(100)
        sess.run(train_step, feed_dict={x: batch_xs, y_: batch_ys})

    # Test trained model
    correct_prediction = tf.equal(y, tf.argmax(y_, 3))
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    test_x, test_y = images.getTestData()
    print(sess.run(accuracy, feed_dict={x:test_x , y_:test_y}))

if __name__ == '__main__':
    main()
