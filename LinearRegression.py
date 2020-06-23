from IPython import display
from matplotlib import pyplot as plt
from mxnet import autograd, nd
import random
from mxnet.gluon import data as gdata
from mxnet.gluon import nn
from mxnet import init
from mxnet.gluon import loss as gloss
from mxnet import gluon

def use_svg_display():
    display.set_matplotlib_formats('svg')


def set_figsize(figsize=(3.5, 2.5)):
    use_svg_display()
    plt.rcParams['figure.figsize'] = figsize


def linreg(x, w, b):
    return nd.dot(x, w) + b


def squared_loss(y_hat, y):
    return (y_hat - y.reshape(y_hat.shape))**2 / 2


def sgd(params, lr, batch_size):
    for param in params:
        param[:] = param - lr * param.grad / batch_size


def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        j = nd.array(indices[i: min(i+batch_size, num_examples)])
        yield features.take(j), labels.take(j)


def method0():
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    set_figsize()
    plt.scatter(features[:, 1].asnumpy(), labels.asnumpy(), 1)
    w = nd.random.normal(scale=0.01, shape=(num_inputs, 1))
    b = nd.zeros(shape=(1,))
    w.attach_grad()
    b.attach_grad()
    lr = 0.03
    num_epochs = 13
    loss = squared_loss
    batch_size = 10
    for epoch in range(num_epochs):
        for x, y in data_iter(batch_size, features, labels):
            with autograd.record():
                l = loss(linreg(x, w, b), y)
            l.backward()
            sgd([w, b], lr, batch_size)
        train_l = loss(linreg(features, w, b), labels)
        print('epoch %d, loss %f' % (epoch, train_l.mean().asnumpy()))
    print(w, b)

def method1():
    num_inputs = 2
    num_examples = 1000
    true_w = [2, -3.4]
    true_b = 4.2
    features = nd.random.normal(scale=1, shape=(num_examples, num_inputs))
    labels = true_w[0] * features[:, 0] + true_w[1] * features[:, 1] + true_b
    labels += nd.random.normal(scale=0.01, shape=labels.shape)
    batch_size = 10
    dataset = gdata.ArrayDataset(features, labels)
    data_iters = gdata.DataLoader(dataset, batch_size, shuffle=True)
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize(init.Normal(sigma=0.01))
    loss = gloss.L2Loss()
    trainer = gluon.Trainer(net.collect_params(), 'sgd', {'learning_rate': 0.03})
    num_epochs = 3
    for epoch in range(1, num_epochs+1):
        for x, y in data_iters:
            with autograd.record():
                l = loss(net(x), y)
            l.backward()
            trainer.step(batch_size)
        l = loss(net(features), labels)
        print('epoch: %d, loss: %f' % (epoch, l.mean().asnumpy()))
    dense = net[0]
    print(dense.weight.data())
    print(dense.bias.data())


def main():
    method1()

if __name__ == "__main__":
    main()

