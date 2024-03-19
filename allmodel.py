# -*- coding: utf-8 -*-
"""
Created on Thu Jan 13 17:56:52 2022

@author: dell
"""
# classes
import torch
import torch.utils.data as Data

class lstm(torch.nn.Module):
    def __init__(self, dim, num_class=2):
        super(lstm, self).__init__()
        self.lstm = torch.nn.LSTM(
            input_size=dim,
            hidden_size=128,
            batch_first=True)
        self.fc = torch.nn.Linear(128, num_class)
        self.sf = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to(torch.float32)
        out, (h_0, c_0) = self.lstm(x)
        out = self.fc(out)
        out = self.sf(out)
        return out

class rnn(torch.nn.Module):
    def __init__(self, dim, num_class=2):
        super(rnn, self).__init__()
        self.rnn = torch.nn.RNN(
            input_size=dim,
            hidden_size=128,
            batch_first=True)
        self.fc = torch.nn.Linear(128, num_class)
        self.sf = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = torch.tensor(x, dtype=torch.float32)
        out, h_n = self.rnn(x)
        out = self.fc(out)
        out = self.sf(out)
        return out

class mlp(torch.nn.Module):
    def __init__(self, dim, num_class=2):
        super(mlp, self).__init__()
        self.input = torch.nn.Linear(dim, 128)
        self.fc = torch.nn.Linear(128, num_class)
        self.sf = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.input(x)
        out = self.fc(out)
        out = self.sf(out)
        return out

class adversarial(torch.nn.Module):
    def __init__(self, dim, num_class=2):
        super(adversarial, self).__init__()
        self.input = torch.nn.Linear(dim, 32)
        self.fc = torch.nn.Linear(32, num_class)
        self.sf = torch.nn.Softmax(dim=1)

    def forward(self, x):
        x = x.to(torch.float32)
        out = self.input(x)
        out = self.fc(out)
        out = self.sf(out)
        return out

def demographic_parity_loss(y_pred, sensitive_attr):
    # Calculate the mean predicted probabilities for each group
    group0_prob = torch.mean(y_pred[sensitive_attr == 0][:, 1])
    group1_prob = torch.mean(y_pred[sensitive_attr == 1][:, 1])

    # Calculate the demographic parity constraint
    dp_loss = torch.abs(group0_prob - group1_prob)

    return dp_loss

class DLClassifier():
    def __init__(self, model, lr, minibatch, epoch, verbose=True, cuda=False, reduction=False):
        self.model = model
        self.loss = None
        self.val_loss = None
        self.epoch = epoch
        self.lr = lr
        self.minibatch = minibatch
        self.optimizer = None
        self.scheduler = None
        self.criterion = None
        self.verbose = verbose
        self.cuda = cuda
        self.reduction = reduction

    def fit(self, x_train, y_train, x_test, y_test, s_train=None):
        if s_train:
            torch_dataset = Data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train), torch.from_numpy(s_train))
        else:
            torch_dataset = Data.TensorDataset(torch.from_numpy(x_train), torch.from_numpy(y_train))
        loader = Data.DataLoader(  # 批训练数据加载器
            dataset=torch_dataset,
            batch_size=self.minibatch,
            shuffle=True,  # 每次训练打乱数据， 默认为False
            #        num_workers=2,  # 使用多进行程读取数据， 默认0，为不使用多进程
            drop_last=False
        )

        if self.cuda and torch.cuda.is_available():
            self.model = self.model.cuda()
            self.criterion = torch.nn.CrossEntropyLoss().cuda()
        else:
            self.model = self.model.cpu()
            self.criterion = torch.nn.CrossEntropyLoss()

        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.5, patience=20,
                                                               verbose=False, threshold=0.0001, threshold_mode='rel',
                                                               cooldown=0, min_lr=1e-05, eps=1e-08)

        loss_list = []
        valid_loss_list = []

        for epoch in range(self.epoch):
            self.model.train()
            epoch_loss = 0
            loss_ = 0
            if s_train:
                for idx, (x, target, s) in enumerate(loader, 0):
                    if self.cuda and torch.cuda.is_available():
                        x = x.cuda()
                        target = target.cuda()
                    else:
                        x = x.cpu()
                        target = target.cpu()
                    predict = self.model(x)
                    #            losses.append(loss)
                    self.optimizer.zero_grad()
                    loss = self.criterion(predict, target.long())

                    if self.reduction:
                        dp_loss = demographic_parity_loss(predict, s)
                        loss += dp_loss

                    loss.backward()
                    self.optimizer.step()
                    self.scheduler.step(loss)
                    epoch_loss += loss.item()
                    loss_ = epoch_loss / (idx + 1)
                    del x, target, predict
                    if self.cuda and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                loss_list.append(loss_)
                with torch.no_grad():
                    if self.cuda and torch.cuda.is_available():
                        pred = self.model(torch.tensor(x_test).cuda())
                        y = torch.tensor(y_test).cuda()
                    else:
                        pred = self.model(torch.tensor(x_test))
                        y = torch.tensor(y_test)
                    valid_loss = self.criterion(pred, y.long())
                    valid_loss_list.append(valid_loss.item())
                    if self.verbose:
                        correct = int(torch.sum(torch.argmax(pred, dim=1) == torch.argmax(y, dim=1)))
                        total = len(y)
                        print("Epoch={}/{}, train_loss={}, valid_loss={}, valid_acc={}, lr={}".format(
                            epoch + 1, self.epoch, loss_, valid_loss, correct / total,
                            self.optimizer.state_dict()['param_groups'][0]['lr']))
            else:
                for idx, (x, target) in enumerate(loader, 0):
                    if self.cuda and torch.cuda.is_available():
                        x = x.cuda()
                        target = target.cuda()
                    else:
                        x = x.cpu()
                        target = target.cpu()
                    predict = self.model(x)
                    #            losses.append(loss)
                    self.optimizer.zero_grad()
                    loss = self.criterion(predict, target.long())

                    loss.backward()
                    self.optimizer.step()
                    #self.scheduler.step(loss)
                    epoch_loss += loss.item()
                    loss_ = epoch_loss / (idx + 1)
                    del x, target, predict
                    if self.cuda and torch.cuda.is_available():
                        torch.cuda.empty_cache()
                loss_list.append(loss_)
                with torch.no_grad():
                    if self.cuda and torch.cuda.is_available():
                        pred = self.model(torch.tensor(x_test).cuda())
                        y = torch.tensor(y_test).cuda()
                    else:
                        pred = self.model(torch.tensor(x_test))
                        y = torch.tensor(y_test)
                    valid_loss = self.criterion(pred, y.long())
                    valid_loss_list.append(valid_loss.item())
                    if self.verbose:
                        correct = int(torch.sum(torch.argmax(pred, dim=1) == y))
                        total = len(y)
                        print("Epoch={}/{}, train_loss={}, valid_loss={}, valid_acc={}, lr={}".format(
                            epoch + 1, self.epoch, loss_, valid_loss, correct / total,
                            self.optimizer.state_dict()['param_groups'][0]['lr']))

        #            if valid_loss <= min_loss_val and epoch > 30:
        #                min_loss_val = valid_loss
        #                torch.save(model, os.path.join(root, 'best_model.pth'))
        self.loss = loss_list
        self.val_loss = valid_loss_list
        print('Training finished.')

    def predict(self, X):
        self.model.eval()
        if self.cuda:
            outputs = self.model(torch.tensor(X).cuda())
        else:
            outputs = self.model(torch.tensor(X).cpu())
        pred = torch.argmax(outputs, dim=1).cpu().numpy().astype('int64')

        return pred

    def predict_prob(self, X):
        self.model.eval()
        if self.cuda:
            outputs = self.model(torch.tensor(X).cuda())
        else:
            outputs = self.model(torch.tensor(X).cpu())
        pred = outputs[:, 1].detach().cpu().numpy()

        return pred
    
    
    
    