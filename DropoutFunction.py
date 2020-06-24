import d2lzh as d2l
from mxnet import autograd, gluon, nd, init
from mxnet.gluon import data as gdata, loss as gloss, nn


num_inputs, num_outputs, num_hiddens1, num_hiddens2 = 784, 10, 256, 256
w1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens1))
b1 = nd.zeros(num_hiddens1)
w2 = nd.random.normal(scale=0.01, shape=(num_hiddens1, num_hiddens2))
b2 = nd.zeros(num_hiddens2)
w3 = nd.random.normal(scale=0.01, shape=(num_hiddens2, num_outputs))
b3 = nd.zeros(num_outputs)
params = [w1, b1, w2, b2, w3, b3]
for param in params:
    param.attach_grad()
drop_prob1, drop_prob2 = 0.2, 0.5


def net(x):
    x = x.reshape((-1, num_inputs))
    h1 = (nd.dot(x, w1) + b1).relu()
    if(autograd.is_training()):  # only drop out in trainning mode
        h1 = dropout(h1, drop_prob1)
    h2 = (nd.dot(h1, w2) + b2).relu()
    if (autograd.is_training()):  # only drop out in trainning mode
        h2 = dropout(h2, drop_prob2)
    return nd.dot(h2, w3) + b3


def dropout(x, drop_prob):
    assert 0 <= drop_prob <= 1
    keep_prob = 1 - drop_prob
    if keep_prob == 0:
        # in this situation, all elements are dropped out
        return x.zeros_like()
    mask = nd.random.uniform(0, 1, x.shape) < keep_prob
    return mask * x / keep_prob


def method0():
    num_epochs, lr, batch_size = 5, 0.5, 256
    loss = gloss.SoftmaxCrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
                  params, lr)


def method1():
    num_epochs, lr, batch_size = 5, 0.5, 256
    loss = gloss.SoftmaxCrossEntropyLoss()
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    net = nn.Sequential()
    net.add(nn.Dense(num_hiddens1, activation="relu"),
            nn.Dropout(drop_prob1),
            nn.Dense(num_hiddens2, activation="relu"),
            nn.Dropout(drop_prob2),
            nn.Dense(num_outputs))
    net.initialize(init.Normal(sigma=0.01))
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': lr})
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size, None,
                  None, trainer)


def main():
    #method0()
    method1()
    return


if __name__ == '__main__':
    main()

