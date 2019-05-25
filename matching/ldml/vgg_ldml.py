import torch
import os
import torchvision
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.autograd import Variable
import pandas as pd
import numpy as np
from torch import nn
from torch import optim

working_dir = '/opt/hades/'


batch_size = 64
train_fraction = 0.9
stop_training = 0.1
emb_size = 512
epoches = 10
os.environ["CUDA_VISIBLE_DEVICES"]='1'
class MyDataset(Dataset):

    def __init__(self, data):
        super().__init__()
        self.data = data

    def __getitem__(self, index):
        A, B, label = self.data[index]
        return A, B, label

    def __len__(self):
        return len(self.data)
class EarlyStopping:
    """Early stops the training if validation loss dosen't improve after a given patience."""
    def __init__(self, patience=7, verbose=False):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 7
            verbose (bool): If True, prints a message for each validation loss improvement.
                            Default: False
        """
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf

    def __call__(self, val_loss, model):

        score = - val_loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
        elif score < self.best_score:
            self.counter += 1
            print('EarlyStopping counter: {} out of {}'.format(self.counter, self.patience))
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, model):
        '''Saves model when validation loss decrease.'''
        if self.verbose:
            print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(self.val_loss_min, val_loss))
        torch.save(model.state_dict(), '/opt/hades/checkpoint.ptM_vgg_ada_1e3_sche_100')
        self.val_loss_min = val_loss

class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()
        diag = np.random.rand(emb_size)
        diag += 0.5
        diag = np.diag(diag)
        bias = np.random.rand(1)
        bias += 0.5
        print('init bias: {}'.format(bias))
        # diag = np.linalg.cholesky(diag)

        self.W = nn.Linear(emb_size, emb_size)
        nn.init.normal_(self.W.weight, 0, 0.01)
        nn.init.constant_(self.W.bias, 0)

        self.M = nn.Parameter(torch.tensor(diag, requires_grad=True).float())
        self.b = nn.Parameter(torch.tensor(bias, requires_grad=True).float())


    def forward(self, *input):
        A, B = input
        # M = torch.mm(self.M.t(), self.M)
        A = self.W(A)
        B = self.W(B)
        x = (A - B) # batch_size x emb_dims
        xM = torch.mm(x, self.M) # batch_size x emb_dims
        # xM = torch.mm(x, M)
        d = torch.bmm(xM.view(xM.size()[0], 1, xM.size()[1]), x.view(x.size()[0], x.size()[1], 1))
        # dd = d.clone().cpu().detach().numpy()
        # assert (dd > 0).all(), dd
        # sigmoid = torch.sigmoid(self.b - d)
        # return sigmoid
        return self.b - d

train_ids = []
val_ids = []
te = os.listdir( working_dir + 'vgg_face2/test_features/')
for idt in te:
    train_ids.append(np.load(working_dir + 'vgg_face2/test_features/' + idt))
va = os.listdir( working_dir + 'vgg_face2/valid_features/')
for idt in va:
    val_ids.append(np.load(working_dir + 'vgg_face2/valid_features/' + idt))
tr = os.listdir(working_dir + 'vgg_face2/train_features/')
for idt in tr:
    train_ids.append(np.load(working_dir + 'vgg_face2/train_features/' + idt))

train_povs = []
train_negs = []
val_povs = []
val_negs = []

def construct_pairs(ids_list, data):
    povs = []
    negs = []
    ratio = 0.01 if data == 'train' else 0.02

    for i in range(len(ids_list) - 1):
        for j in range(i + 1, len(ids_list)):
            for f1 in ids_list[i]:
                for f2 in ids_list[j]:
                    if np.random.rand() < ratio:
                        negs.append((f1, f2, 0))
    for idt in ids_list:
        for i1 in range(len(idt) - 1):
            for i2 in range(i1 + 1, len(idt)):
                povs.append((idt[i1], idt[i2], 1))

    print('povs', len(povs))
    print('negs', len(negs))
    return povs + negs


# prepare data
data_train = np.array(construct_pairs(train_ids, 'train'))
print('povs in train: {}'.format(sum(data_train[:, 2])))
print('negs in train: {}'.format(len(data_train) - sum(data_train[:, 2])))

data_valid = np.array(construct_pairs(val_ids, 'val'))
print('povs in valid: {}'.format(sum(data_valid[:, 2])))
print('negs in valid: {}'.format(len(data_valid) - sum(data_valid[:, 2])))

dataset_train = MyDataset(data_train)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_valid = MyDataset(data_valid)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size)

# prepare model
model = MyModel()
model.cuda()
crit = nn.BCEWithLogitsLoss(reduction='none') # reduction is none for OHEM purpose
# optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.1) # TODO: more settings
optimizer = torch.optim.Adagrad(model.parameters(), lr=0.001)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=1000)
# sanity test
# x, y, label = next(iter(dataloader))
# x, y, label = x.cuda(), y.cuda(), label.cuda().double()
# print(label)
# pred = model(x, y).double()
# label = label.view(batch_size, -1)
# loss = crit(pred, label)
# print(loss)
# print(model.M)
# optimizer.zero_grad()
# loss.backward()
# optimizer.step()
# pred = model(x, y).double()
# label = label.view(batch_size, -1)
# loss = crit(pred, label)
# print(loss)
# print(model.M)
#
# exit()


def evaluate(model, data):

    if data == 'train':
        dataloader = dataloader_train
    elif data == 'val':
        dataloader = dataloader_valid
    else:
        dataloader = None
    count_right = 1
    count_wrong = 0
    neg_dists = []
    pov_dists = []
    neg_dists_r = []
    pov_dists_r = []

    for A, B, label in dataloader:
        model.eval()

        # calculate Mal distance
        A, B, label = A.cuda(), B.cuda(), label.cuda().float()
        label = label.view(-1, )
        pred = model(A, B).float().view(-1, )

        pred_np = pred.clone().cpu().detach().numpy()
        label_np = label.clone().cpu().detach().numpy()
        # calculate original distance
        x = (A-B) # batch_size x emb_size
        x = np.square(torch.norm(x, dim=1).clone().cpu().detach().numpy()) # batch_size x 1
        # save distances for plotting
        bias = model.b.item()
        for i, l in enumerate(label_np):
            if l == 0:
                neg_dists.append(bias - pred_np[i])
                neg_dists_r.append(x[i])
                if pred_np[i] < 0:
                    count_right += 1
                else:
                    count_wrong += 1
            elif l == 1:
                pov_dists.append(bias - pred_np[i])
                pov_dists_r.append(x[i])

                if pred_np[i] >= 0:
                    count_right += 1
                else:
                    count_wrong += 1

    neg_dists = np.array(neg_dists)
    pov_dists = np.array(pov_dists)
    neg_dists_r = np.array(neg_dists_r)
    pov_dists_r = np.array(pov_dists_r)

    prefix = 'vgg_' + data + '_'
    np.save(file='/opt/hades/' + prefix + 'neg_dists_M_ada_1e3_sche_100', arr=neg_dists)
    np.save(file='/opt/hades/' + prefix + 'pov_dists_M_ada_1e3_sche_100', arr=pov_dists)
    np.save(file='/opt/hades/' + prefix + 'neg_dists_rM_ada_1e3_sche_100', arr=neg_dists_r)
    np.save(file='/opt/hades/' + prefix + 'pov_dists_rM_ada_1e3_sche_100', arr=pov_dists_r)
    print('rights in {}: {}'.format(data, count_right))
    print('wrongs in {}: {}'.format(data, count_wrong))
    return count_wrong / count_right

def train():
    count = 0
    early_stopping = EarlyStopping(patience=100, verbose=True)
    train_loss = []
    for e in range(epoches):
        for A, B, label in dataloader_train:
            count += 1
            if count % 1000 == 0:

                print(model.b)
                valid_loss = evaluate(model, 'val')

                # early stopping
                early_stopping(valid_loss, model)
                if early_stopping.early_stop:
                    print("Early stopping")
                    # load the last checkpoint with the best model
                    model.load_state_dict(torch.load('/opt/hades/checkpoint.ptM_vgg_ada_1e3_sche_100'))
                    return

                if valid_loss < stop_training:
                    return
                model.train()
            else:
                A, B, label = A.cuda(), B.cuda(), label.cuda().float()

                label = label.view(-1,)
                pred = model(A, B).float().view(-1,)
                loss = crit(pred, label) # reduce=False for crit
                # OHEM
                indices = torch.topk(loss, int(0.5 * loss.size()[0]))
                ohem_loss = torch.mean(loss[indices[1]])
                train_loss.append(ohem_loss.item())
                # print(np.mean(train_loss))
                optimizer.zero_grad()
                # loss = torch.mean(loss)
                # loss.backward()
                ohem_loss.backward()
                optimizer.step()
                # scheduler.step()
train()
print(model.M)
print(model.b)
evaluate(model, 'train')
evaluate(model, 'val')