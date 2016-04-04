import tensorflow as tf


def load_mnist_data():
    """
    Loads tutorial data.
    :return:
    """
    from tensorflow.examples.tutorials.mnist import input_data
    mnist = input_data.read_data_sets('MNIST_data', one_hot=True)
    return mnist


def softmax_regression_model():
    """
    Simplistic example of TF using a regression model
    :return:
    """

    # training data
    mnist = load_mnist_data()

    sess = tf.InteractiveSession()

    # graph setup
    x = tf.placeholder(tf.float32, shape=[None, 784])
    y_ = tf.placeholder(tf.float32, shape=[None, 10])
    W = tf.Variable(tf.zeros([784, 10]))
    b = tf.Variable(tf.zeros([10]))

    # initialize
    sess.run(tf.initialize_all_variables())

    # model setup
    y = tf.nn.softmax(tf.matmul(x, W) + b)
    cross_entropy = -tf.reduce_sum(y_ * tf.log(y))

    # training
    train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
    for i in range(1000):
        batch = mnist.train.next_batch(50)
        train_step.run(feed_dict={x: batch[0], y_: batch[1]})

    # check
    correct_prediction = tf.equal(tf.argmax(y, 1), tf.argmax(y_, 1))  # list of bools
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))  # converts list of bools to acc
    print(accuracy.eval(feed_dict={x: mnist.test.images, y_: mnist.test.labels}))


def hello_world():
    import tensorflow
    hello = tensorflow.constant('Hello, Tensorflow!')
    sess = tensorflow.Session()
    print sess.run(hello)
    a = tensorflow.constant(10)
    b = tensorflow.constant(32)
    print sess.run(a + b)


if __name__ == '__main__':
    softmax_regression_model()
