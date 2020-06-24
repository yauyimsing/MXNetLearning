import d2lzh as d2l
from mxnet.gluon import data as gdata
import sys
import time


def main():
    mnist_train = gdata.vision.FashionMNIST(train=True)
    mnist_test = gdata.vision.FashionMNIST(train=False)
    train_l = len(mnist_train)
    test_l = len(mnist_test)
    print(train_l, test_l)
    feature, labl = mnist_train[0]
    # print(feature)
    x, y = mnist_train[0:9]
    show_fashion_mnist(x, get_fashion_mnist_labels(y))


def get_fashion_mnist_labels(labels):
    text_labels = ['t-shirt', 'trouser', 'pullover', 'dress', 'coat',
                   'sandal', 'shirt', 'sneaker', 'bag', 'ankle boot']
    return [text_labels[int(i)] for i in labels]


def show_fashion_mnist(images, labels):
    d2l.use_svg_display()
    _, figs = d2l.plt.subplots(1, len(images), figsize=(12, 12))
    for f, img, lbl in zip(figs, images, labels):
        f.imshow(img.reshape((28, 28)).asnumpy())
        f.set_title(lbl)
        f.axes.get_xaxis().set_visible(False)
        f.axes.get_yaxis().set_visible(False)
        print('show')

if __name__ == "__main__":
    main()

