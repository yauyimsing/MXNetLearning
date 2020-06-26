from mxnet import nd, init
from mxnet.gluon import nn

class MLP(nn.Block):
    # initialize model layer parameters, with two full connected layer
    def __init__(self, **kwargs):
        super(MLP, self).__init__(**kwargs)
        self.hidden = nn.Dense(256, activation='relu')
        self.output = nn.Dense(10)

    # model forward calculation
    def forward(self, x):
        return self.output(self.hidden(x))


class MySequential(nn.Block):
    def __init__(self, **kwargs):
        super(MySequential, self).__init__(**kwargs)

    def add(self, block):
        self._children[block.name] = block

    def forward(self, x):
        for block in self._children.values():
            x = block(x)
        return x


class FancyMLP(nn.Block):
    def __init__(self, **kwargs):
        super(FancyMLP, self).__init__(**kwargs)
        self.rand_weight = self.params.get_constant(
            'rand_weight', nd.random.uniform(shape=(20, 20)))
        self.dense = nn.Dense(20, activation='relu')

    def forward(self, x):
        x = self.dense(x)
        x = nd.relu(nd.dot(x, self.rand_weight.data()) + 1)
        x = self.dense(x)
        while (x.norm().asscalar() > 1):
            t = x.norm().asscalar()
            # print('t', t)
            x /= 2
        if x.norm().asscalar() < 0.8:
            x *= 10
        return x.sum()


class NestMLP(nn.Block):
    def __init__(self, **kwargs):
        super(NestMLP, self).__init__(**kwargs)
        self.net = nn.Sequential()
        self.net.add(nn.Dense(64, activation='relu'),
                     nn.Dense(32, activation='relu'))
        self.dense = nn.Dense(16, activation='relu')

    def forward(self, x):
        return self.dense(self.net(x))


class MyInit(init.Initializer):
    def _init_weight(self, name, data):
        print("Init", name, data.shape)
        data[:] = nd.random.uniform(low=-10, high=10, shape=data.shape)
        data *= data.abs() >= 5


class CneteredLayer(nn.Block):
    def __init__(self, **kwargs):
        super(CneteredLayer, self).__init__(**kwargs)

    def forward(self, x):
        return x - x.mean()


class MyDense(nn.Block):
    # units is the number of output, in_units is the number of input
    def __init__(self, units, in_units, **kwargs):
        super(MyDense, self).__init__(**kwargs)
        self.weight = self.params.get('weight', shape=(in_units, units))
        self.bias = self.params.get('bias', shape=(units,))

    def forward(self, x):
        # print('x', x)
        # print('weight', self.weight.data())
        # print('bias', self.bias)
        linear = nd.dot(x, self.weight.data()) + self.bias.data()
        return nd.relu(linear)


def Method0():
    x = nd.random.uniform(shape=(2, 20))
    net = MLP()
    net.initialize()
    o = net(x)
    print(o)


def Method1():
    x = nd.random.uniform(shape=(2, 20))
    net = MySequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    net.initialize()
    o = net(x)
    print(o)


def Method2():
    x = nd.random.uniform(shape=(2, 20))
    net = FancyMLP()
    net.initialize()
    o = net(x)
    print(o)


def Method3():
    x = nd.random.uniform(shape=(2, 20))
    net = nn.Sequential()
    net.add(NestMLP(), nn.Dense(20), FancyMLP())
    net.initialize()
    o = net(x)
    print(o)


def Method4():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'))
    net.add(nn.Dense(10))
    #net.initialize()
    net.initialize(MyInit(), force_reinit=True)
    x = nd.random.uniform(shape=(2, 20))
    y = net(x)
    net[0].weight.data()[0]


def Method5():
    net = nn.Sequential()
    shared = nn.Dense(8, activation='relu')
    net.add(nn.Dense(8, activation='relu'),
            shared,
            nn.Dense(8, activation='relu', params=shared.params),
            nn.Dense(10))
    net.initialize()
    x = nd.random.uniform(shape=(2, 20))
    net(x)
    print(net[1].weight.data()[0] == net[2].weight.data()[0])


def Method6():
    net = nn.Sequential()
    net.add(nn.Dense(256, activation='relu'),
            nn.Dense(10))
    net.initialize(init=MyInit()) # delay initialization
    x = nd.random.uniform(shape=(2, 20))
    y = net(x)


def Method7():
    layer = CneteredLayer()
    y = layer(nd.array([1, 2, 3, 4, 5]))
    print(y)
    net = nn.Sequential()
    net.add(nn.Dense(128), CneteredLayer())
    net.initialize()
    y = net(nd.random.uniform(shape=(4, 8)))
    y1 = y.mean().asscalar()
    print(y1)


def Method8():
    dense = MyDense(units=3, in_units=5)
    print(dense.params)
    dense.initialize()
    y = dense(nd.random.uniform(shape=(2, 5)))
    print(y)
    net = nn.Sequential()
    net.add(MyDense(8, in_units=64))
    net.add(MyDense(1, in_units=8))
    net.initialize()
    y = net(nd.random.uniform(shape=(2, 64)))
    print(y)


def main():
    # Method0()
    # Method1()
    # Method2()
    # Method3()
    # Method4()
    # Method5()
    # Method6()
    # Method7()
    Method8()


if __name__=='__main__':
    main()
