import d2lzh as d2l
from mxnet import nd, gluon, init
from mxnet.gluon import loss as gloss, nn


batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs, num_outputs, num_hiddens = 784, 10, 256
w1 = nd.random.normal(scale=0.01, shape=(num_inputs, num_hiddens))
b1 = nd.zeros(num_hiddens)
w2 = nd.random.normal(scale=0.01, shape=(num_hiddens, num_outputs))
b2 = nd.zeros(num_outputs)


def relu(x):
    return nd.maximum(x, 0)


def net(x):
    x = x.reshape((-1, num_inputs))
    h = relu(nd.dot(x, w1) + b1)
    return nd.dot(h, w2) + b2


def Method0():
    params = [w1, b1, w2, b2]
    for param in params:
        param.attach_grad()
    loss = gloss.SoftmaxCrossEntropyLoss()
    num_epochs, lr = 5, 0.5
    d2l.train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
                  params, lr)
    for x, y in test_iter:
        break
    true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
    pre_labels = d2l.get_fashion_mnist_labels(net(x).argmax(axis=1).asnumpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pre_labels)]
    d2l.show_fashion_mnist(x[0:9], titles[0:9])


def Method1():
    vnet = nn.Sequential()
    vnet.add(nn.Dense(num_hiddens, activation='relu'),
            nn.Dense(num_outputs))
    vnet.initialize(init.Normal(sigma=0.01))
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(vnet.collect_params(), 'sgd', {'learning_rate': 0.05})
    num_epochs = 50
    d2l.train_ch3(vnet, train_iter, test_iter, loss, num_epochs, batch_size, None,
                  None, trainer)
    for x, y in test_iter:
        break
    true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
    pre_labels = d2l.get_fashion_mnist_labels(vnet(x).argmax(axis=1).asnumpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pre_labels)]
    d2l.show_fashion_mnist(x[0:9], titles[0:9])



def main():
    #Method0()
    Method1()


if __name__ == "__main__":
    main()