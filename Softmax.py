import d2lzh as d2l
from mxnet import autograd, nd, gluon, init
from mxnet.gluon import loss as gloss, nn

batch_size = 256
train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
num_inputs = 28 * 28
num_outputs = 10
w = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
b = nd.zeros(num_outputs)
w.attach_grad()
b.attach_grad()


def net(x, w, b, num_inputs):
    t0 = x.reshape((-1, num_inputs))
    t1 = nd.dot(t0, w)
    t2 = t1 + b
    t3 = softmax(t2)
    return t3


def accuracy(y_hat, y):
    return (y_hat.argmax(axis=1) == y.astype('float32')).mean().asscalar()


def evaluate_accuracy(data_iter, netFunc, w, b, num_inputs):
    acc_sum, n = 0.0, 0
    i = 0
    for x, y in data_iter:
        i += 1
        #print(i, y)
        y = y.astype('float32')
        acc_sum += (netFunc(x, w, b, num_inputs).argmax(axis=1) == y).sum().asscalar()
        n += y.size
    return acc_sum / n


def softmax(x):
    x_exp = x.exp()
    partition = x_exp.sum(axis=1, keepdims=True)
    return x_exp / partition


def train_ch3(net, w, b, num_inputs, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, trainer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for x, y in train_iter:
            with autograd.record():
                y_hat = net(x, w, b, num_inputs)
                l = loss(y_hat, y).sum()
            l.backward()
            if trainer is None:
                d2l.sgd(params, lr, batch_size)
            else:
                trainer.step(batch_size)
            y = y.astype('float32')
            train_l_sum += l.asscalar()
            train_acc_sum += (y_hat.argmax(axis=1) == y).sum().asscalar()
            n += y.size
        test_acc = evaluate_accuracy(test_iter, net, w, b, num_inputs)
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch, train_l_sum/n, train_acc_sum / n, test_acc))


def cross_entropy(y_hat, y):
    return -nd.pick(y_hat, y).log()


def method0():
    #batch_size = 256
    #train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size)
    #num_inputs = 28 * 28
    #num_outputs = 10
    #w = nd.random.normal(scale=0.01, shape=(num_inputs, num_outputs))
    #b = nd.zeros(num_outputs)
    #w.attach_grad()
    #b.attach_grad()
    c = evaluate_accuracy(train_iter, net, w, b, num_inputs)
    # print(c)
    num_epochs, lr = 5, 0.1
    train_ch3(net, w, b, num_inputs, train_iter, test_iter, cross_entropy, num_epochs, batch_size,
              [w, b], lr)
    for x, y in test_iter:
        break
    true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
    pre_labels = d2l.get_fashion_mnist_labels(net(x, w, b, num_inputs).argmax(axis=1).asnumpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pre_labels)]
    d2l.show_fashion_mnist(x[0:9], titles[0:9])


def method1():
    vnet = nn.Sequential()
    vnet.add(nn.Dense(10))
    vnet.initialize(init.Normal(sigma=0.01))
    loss = gloss.SoftmaxCrossEntropyLoss()
    trainer = gluon.Trainer(vnet.collect_params(), 'sgd', {'learning_rate': 0.1})
    num_epochs = 15
    d2l.train_ch3(vnet, train_iter, test_iter, loss, num_epochs, batch_size, None, None, trainer)
    for x, y in test_iter:
        break
    true_labels = d2l.get_fashion_mnist_labels(y.asnumpy())
    pre_labels = d2l.get_fashion_mnist_labels(vnet(x).argmax(axis=1).asnumpy())
    titles = [true + '\n' + pred for true, pred in zip(true_labels, pre_labels)]
    d2l.show_fashion_mnist(x[0:9], titles[0:9])


def main():
    print('main start')
    method1()


if __name__ == '__main__':
    main()

