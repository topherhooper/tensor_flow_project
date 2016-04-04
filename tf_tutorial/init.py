import tensorflow


def hello_world():
    hello = tensorflow.constant('Hello, Tensorflow!')
    sess = tensorflow.Session()
    print sess.run(hello)
    a = tensorflow.constant(10)
    b = tensorflow.constant(32)
    print  sess.run(a + b)


if __name__ == '__main__':
    hello_world()
