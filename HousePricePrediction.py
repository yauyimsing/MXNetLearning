import d2lzh as d2l
from mxnet import autograd, gluon, init, nd
from mxnet.gluon import data as gdata, loss as gloss, nn
import numpy as np
import pandas as pd

train_data = pd.read_csv('./data/train.csv')
test_data = pd.read_csv('./data/test.csv')
train_data.iloc[0:4, [0, 1, 2, 3, -3, -2, -1]]
all_features = pd.concat((train_data.iloc[:, 1:-1], test_data.iloc[:, 1:]))
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))
all_features = all_features.fillna(all_features.mean())
all_features = pd.get_dummies(all_features, dummy_na=True)
print(all_features.shape)
n_train = train_data.shape[0]
print(n_train)
train_features = nd.array(all_features[:n_train].values)
test_features = nd.array(all_features[n_train:].values)
train_labels = nd.array(train_data.SalePrice.values).reshape((-1, 1))
loss = gloss.L2Loss()



def main():
    k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    train_l, valid_l = k_fold(k, train_features, train_labels,
                              num_epochs, lr, weight_decay, batch_size)
    print('%d-fold validatioin: avg train rmse: %f, avg valid rmse: %f'
          % (k, train_l, valid_l))
    train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size)
    return


def get_net():
    net = nn.Sequential()
    net.add(nn.Dense(1))
    net.initialize()
    return net


def log_rmse(net, features, labels):
    clipped_preds = nd.clip(net(features), 1, float('inf'))
    rmse = nd.sqrt(2*loss(clipped_preds.log(), labels.log()).mean())
    return rmse.asscalar()


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gdata.DataLoader(gdata.ArrayDataset(train_features, train_labels),
                                  batch_size, shuffle=True)
    trainer = gluon.Trainer(net.collect_params(), 'adam',
                            {'learning_rate': learning_rate, 'wd': weight_decay})
    for epoch in range(num_epochs):
        for x, y in train_iter:
            with autograd.record():
                l = loss(net(x), y)
            l.backward()
            trainer.step(batch_size)
        train_ls.append(log_rmse(net, train_features, train_labels))
        if(test_labels is not None):
            test_ls.append(log_rmse(net, test_features, test_labels))
    return train_ls, test_ls


def get_k_fold_data(k, i, x, y):
    # k, num_epochs, lr, weight_decay, batch_size = 5, 100, 5, 0, 64
    # i: 0 -> k-1
    # x: x_train, y: y_train
    assert k > 1
    fold_size = x.shape[0] // k
    # fold_size = 1460
    # print('fold_size:', fold_size)
    x_train, y_train = None, None
    for j in range(k):
        # j: 0 -> k-1(k == 5)
        idx = slice(j * fold_size, (j + 1) * fold_size)
        #print('idx', j, idx)
        x_part, y_part = x[idx, :], y[idx]
        if j == i:
            x_valid, y_valid = x_part, y_part
        elif x_train is None:
            x_train, y_train = x_part, y_part
        else:
            x_train = nd.concat(x_train, x_part, dim=0)
            y_train = nd.concat(y_train, y_part, dim=0)
    return x_train, y_train, x_valid, y_valid


def k_fold(k, x_train, y_train, num_epochs,
           learning_rate, weight_decay, batch_size):
    train_l_sum, valid_l_sum = 0, 0
    for i in range(k): # k == 5
        data = get_k_fold_data(k, i, x_train, y_train)
        net = get_net()
        train_ls, valid_ls = train(net, *data, num_epochs, learning_rate,
                                   weight_decay, batch_size)
        train_l_sum += train_ls[-1]
        valid_l_sum += valid_ls[-1]
        if i == 0:
            d2l.semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'rmse',
                         range(1, num_epochs+1), valid_ls,
                         ['train', 'valid'])
        print('fold %d, train rmse: %f, valid rmse: %f'
              % (i, train_ls[-1], valid_ls[-1]))
    return train_l_sum / k, valid_l_sum / k


def train_and_pred(train_features, test_features, train_labels, test_data,
                   num_epochs, lr, weight_decay, batch_size):
    net = get_net()
    train_ls, _ = train(net, train_features, train_labels, None, None,
                        num_epochs, lr, weight_decay, batch_size)
    d2l.semilogy(range(1, num_epochs+1), train_ls, 'epochs', 'rmse')
    print('train rmse %f' % train_ls[-1])
    preds = net(test_features).asnumpy()
    print('preds', test_features.shape, preds, preds.shape)
    test_data['SalePrice'] = pd.Series(preds.reshape(1, -1)[0])
    submission = pd.concat([test_data['Id'], test_data['SalePrice']], axis=1)
    submission.to_csv('submission.csv', index=False)


if __name__ == '__main__':
    main()



