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
office3 = [[0,4,7,11,15,16,17,18], [1,5,8,9,14,22], [2,3,6,10,12,13,19,20,23,24,25]]
taser1 = [[0,2,3,5], [1,4,6]]
taser2 = [[0,1,3,8,10,14,16], [2,12,13,15,17], [4,7,9], [18], [5,6,11]]
reporters = [[0,1,3,7,14,15,19], [2,10,13,16,20,22], [4,8,9,17], [5,6,11,12,18]]
lights = [[0,2], [1,3,4,5,6,7,8]]
axonvn = [[0], [1,6], [2,4,8], [5], [7]]
drunk = [[0,5,19], [3,4,17,18,20,21], [6,7,9], [8,10,11,12], [14], [13,15,16]]
monkey = [[0,3,4,12,15,21,22], [1,8,16,19], [2,9]]
camden = [[0,8,19,26,40], [1,5,11,16,17,22,25], [2,12,14,23,28,34],
          [4,9,15,18,21,24,27,33], [6],[35,38,39], [20,29,30,31], [32,44]]
wounded = [[0,6,24,34,35,39,45,65,71], [1,5,12,17], [3,13,18,19], [4,23], [7,64],
           [30], [40,42,50,56,61,67]]
shooting = [[1,4,11,13,14,15,17,22,23,24,32,38,39,40], [2,12,18,19,25,26,27,29],
            [3,6,9,10,20,21,30,35,37], [28,33,34]]
fleeing = [[7,15], [4,6,9,12,13,14,16], [5,10,11], [3,8]]
blue = [[0,6], [1,2,5,8,9,14,26], [4,16,17,22,24,27], [15,18,21,23]]
store3 = [[0,4,9], [1,3,5,6,11]]
fleeing2 = [[0,4,15,22,31,59,63,83,85,86,92,99], [1,7,12,18,29,38,67],
            [2,3,19,27,34,43,58,65], [8,11,17,20,26,73,79,84]]
vgg_train = os.listdir(working_dir + 'vgg_face2/train_features/')
vgg_train = [[x] for x in vgg_train]
vgg_test = os.listdir(working_dir + 'vgg_face2/test_features/')
vgg_test = [[x] for x in vgg_test]
vgg_valid = os.listdir(working_dir + 'vgg_face2/valid_features/')
vgg_valid = [[x] for x in vgg_valid]

datasets = dict()
dir_to_gt = {'office3_features/': office3,
             'taser1_features/': taser1,
             'taser2_features/': taser2,
             'reporters_features/': reporters,
             'lights_features/': lights,
             'axonvn_features/': axonvn,
             'drunk_features/': drunk,
             'monkey_features/': monkey,
             'camden_features/':camden,
             'wounded_features/':wounded,
             'shooting_features/': shooting,
             'fleeing_features/': fleeing,
             'blue_features/': blue,
             'store3_features/': store3,
             'fleeing2_features/': fleeing2,
             'vgg_face2/train_features/': vgg_train,
             'vgg_face2/test_features/': vgg_test,
             'vgg_face2/valid_features/': vgg_valid}

# face_dir = 'office3/'
# face_features_dir = 'office3_features/'


lr = 0.001
batch_size = 64
train_fraction = 0.9
stop_training = 0.1
emb_size = 512
epoches = 20
min_per_id_pov = 0
max_per_id_pov = 3000
min_per_id_neg = 0
max_per_id_neg = 1500


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
        torch.save(model.state_dict(), '/opt/hades/checkpoint.ptM_mixed_own_rms_1e2')
        self.val_loss_min = val_loss

class MyModel(nn.Module):
    def __init__(self, matrix=None):
        super(MyModel, self).__init__()
        if matrix is not None:
            diag = matrix
        else:
            diag = np.random.rand(emb_size)
            diag += 0.5
            diag = np.diag(diag)
        bias = np.random.rand(1)
        bias += 0.5
        print('init bias: {}'.format(bias))
        print(diag)
        # diag = np.linalg.cholesky(diag)

        # self.W = nn.Linear(emb_size, emb_size)
        # nn.init.normal_(self.W.weight, 0, 0.01)
        # nn.init.constant_(self.W.bias, 0)

        self.M = nn.Parameter(torch.tensor(diag, requires_grad=True).float())
        self.b = nn.Parameter(torch.tensor(bias, requires_grad=True).float())


    def forward(self, *input):
        A, B = input
        # M = torch.mm(self.M.t(), self.M)
        # A = self.W(A)
        # B = self.W(B)
        x = (A - B) # batch_size x emb_dims
        xM = torch.mm(x, self.M) # batch_size x emb_dims
        # xM = torch.mm(x, M)
        d = torch.bmm(xM.view(xM.size()[0], 1, xM.size()[1]), x.view(x.size()[0], x.size()[1], 1))
        # dd = d.clone().cpu().detach().numpy()
        # assert (dd > 0).all(), dd
        # sigmoid = torch.sigmoid(self.b - d)
        # return sigmoid
        return self.b - d

def get_features_dict(vid, features_dir):
    features_dict = dict()
    for track in sum(vid, []):
        features = np.load(working_dir + features_dir + str(track) + ('.npy'
                           if 'vgg' not in features_dir else ''))
        features_dict[track] = features
    return features_dict

def construct_pairs(vid, features_dir):
    features_dict = get_features_dict(vid, features_dir)
    povs = []
    discard_pov = 0
    discard_neg = 0
    for idt in vid:
        for i1, t1 in enumerate(idt):
            for i2, t2 in enumerate(idt):
                if (i1 < i2) or ('vgg' in features_dir):
                    for i in range(len(features_dict[t1])):
                        for j in range(len(features_dict[t2])):
                            if np.linalg.norm(features_dict[t1][i] - features_dict[t2][j]) < 0.5:
                                discard_pov += 1
                                continue
                            if 'vgg' in features_dir:
                                # if np.linalg.norm(features_dict[t1][i] - features_dict[t2][j]) > 1.1:
                                # if np.random.rand() < 0.5:
                                povs.append((features_dict[t1][i], features_dict[t2][j], 1.0))
                            else:
                                povs.append((features_dict[t1][i], features_dict[t2][j], 1.0))
    negs = []
    for i1 in range(len(vid) - 1):
        for i2 in range(i1+1, len(vid)):
            for t1 in vid[i1]:
                for t2 in vid[i2]:
                    for i in range(len(features_dict[t1])):
                        for j in range(len(features_dict[t2])):
                            if np.linalg.norm(features_dict[t1][i] - features_dict[t2][j]) > 2.5:
                                discard_neg += 1
                                continue
                            if 'vgg' in features_dir:
                                # if np.linalg.norm(features_dict[t1][i] - features_dict[t2][j]) < 1.0:
                                if np.random.rand() < 0.04:
                                    negs.append((features_dict[t1][i], features_dict[t2][j], 0.0))
                            else:
                                negs.append((features_dict[t1][i], features_dict[t2][j], 0.0))
    # ids = np.random.randint(0, len(negs), int(len(povs)))
    # negs = [negs[id] for id in ids]
    # if len(povs) > len(negs):
    #     ids = np.random.randint(0, len(negs), int(len(povs)))
    #     negs = [negs[id] for id in ids]

    print('discard negs: {}'.format(discard_neg))
    print('discard povs: {}'.format(discard_pov))
    print('negative samples for {}: {}'.format(features_dir, len(negs)))
    print('positive samples for {}: {}'.format(features_dir, len(povs)))
    datasets[features_dir] = povs + negs
    return povs + negs

def construct_pairs_from_identities(vid, features_dir):
    features_dict = get_features_dict(vid, features_dir)
    povs = []
    negs = []
    for id1, idt in enumerate(vid):
        # construct povs
        pov = []
        for i1, t1 in enumerate(idt):
            for i2, t2 in enumerate(idt):
                if (i1 < i2) or ('vgg' in features_dir):
                    for i in range(len(features_dict[t1])):
                        for j in range(len(features_dict[t2])):
                            if np.linalg.norm(features_dict[t1][i] - features_dict[t2][j]) < 1e-3:
                                assert 'vgg' in features_dir, (features_dict[t1][i], features_dict[t2][j])
                            pov.append((features_dict[t1][i], features_dict[t2][j], 1))

        if len(pov) == 0:
            print('identity has one track: {}'.format(idt))
        else:
            if len(pov) > max_per_id_pov:
                ids = np.random.randint(0, len(pov), max_per_id_pov)
                pov = [pov[i] for i in ids]
            elif len(pov) < min_per_id_pov:
                ids = np.random.randint(0, len(pov), min_per_id_pov)
                pov = [pov[i] for i in ids]
            povs.extend(pov)
        # construct negs
        neg = []
        for id2 in range(len(vid)):
            if id1 is id2:
                continue
            for t1 in vid[id1]:
                for t2 in vid[id2]:
                    for i in range(len(features_dict[t1])):
                        for j in range(len(features_dict[t2])):
                            neg.append((features_dict[t1][i], features_dict[t2][j], 0.0))

        if len(neg) > max_per_id_neg:
            ids = np.random.randint(0, len(neg), max_per_id_neg)
            neg = [neg[i] for i in ids]
        elif len(neg) < min_per_id_neg:
            ids = np.random.randint(0, len(neg), min_per_id_neg)
            neg = [neg[i] for i in ids]
        negs.extend(neg)

    print('negative samples for {}: {}'.format(features_dir, len(negs)))
    print('positive samples for {}: {}'.format(features_dir, len(povs)))

    datasets[features_dir] = povs + negs

for k, v in dir_to_gt.items():
    construct_pairs_from_identities(v, k)


# prepare data
# data_train = np.array(construct_pairs(reporters, 'reporters_features/') +
#                       construct_pairs(drunk, 'drunk_features/') +
#                       construct_pairs(vgg_train, 'vgg_face2/train_features/'))
data_train = np.array(datasets['reporters_features/']
                      + datasets['drunk_features/']
                      + datasets['shooting_features/']
                      + datasets['wounded_features/']
                      + datasets['camden_features/']
                      + datasets['blue_features/']
                      + datasets['fleeing2_features/'])
# data_train = np.empty((0,3))
ids_choosen_test = np.random.randint(0, len(datasets['vgg_face2/test_features/']),
                                     len(datasets['vgg_face2/test_features/'])//2)
datasets['vgg_face2/test_features/'] = [datasets['vgg_face2/test_features/'][i] for i in ids_choosen_test]

print('vgg samples in train: {}'.format(len(datasets['vgg_face2/test_features/'])))
print('own samples in train: {}'.format(len(data_train)))
data_train = np.concatenate((data_train, np.array(datasets['vgg_face2/test_features/'])))
print('povs in train: {}'.format(sum(data_train[:, 2])))
print('negs in train: {}'.format(len(data_train) - sum(data_train[:, 2])))

# calculate eig for training data
samples = []
def get_dict(vid, features_dir):
    feature_dict = get_features_dict(vid, features_dir)
    for k, v in feature_dict.items():
        if 'vgg' not in features_dir:
            print(k, v.shape)
            samples.append(v)

for k, v in dir_to_gt.items():
    get_dict(v, k)
samples = np.concatenate(samples)
# samples = np.concatenate((data_train[:, 0], data_train[:, 1]), axis=0)
print(samples.shape)
cov = np.cov(samples, rowvar=False)
print(cov.shape)
print(cov)
# w, v = np.linalg.eig(cov)
# idx = w.argsort()[::-1]
# w = w[idx]
# v = v[:, idx]
# fh = v[:, :v.shape[1]//2]

# data_valid = np.array(construct_pairs(taser1, 'taser1_features/') +
#                       construct_pairs(taser2, 'taser2_features/') +
#                       construct_pairs(axonvn, 'axonvn_features/') +
#                       construct_pairs(vgg_valid, 'vgg_face2/valid_features/'))
data_valid = np.array(datasets['taser1_features/']
                      + datasets['taser2_features/']
                      + datasets['axonvn_features/']
                      + datasets['lights_features/']
                      + datasets['monkey_features/']
                      + datasets['fleeing_features/']
                      + datasets['store3_features/'])
# data_valid = np.empty((0,3))
ids_choosen_valid = np.random.randint(0, len(datasets['vgg_face2/valid_features/']),
                                      len(datasets['vgg_face2/valid_features/'])//4)
datasets['vgg_face2/valid_features/'] = [datasets['vgg_face2/valid_features/'][i] for i in ids_choosen_valid]
print('vgg samples in valid: {}'.format(len(datasets['vgg_face2/valid_features/'])))
print('own samples in valid: {}'.format(len(data_valid)))
# data_valid = np.concatenate((data_valid, np.array(datasets['vgg_face2/valid_features/'])))
print('povs in valid: {}'.format(sum(data_valid[:, 2])))
print('negs in valid: {}'.format(len(data_valid) - sum(data_valid[:, 2])))
# data_test = np.array(construct_pairs(lights, 'lights_features/') +
#                      construct_pairs(office3, 'office3_features/') +
#                      construct_pairs(vgg_test, 'vgg_face2/test_features/'))
# print('povs in test: {}'.format(sum(data_test[:, 2])))
# print('negs in test: {}'.format(len(data_test) - sum(data_test[:, 2])))

dataset_train = MyDataset(data_train)
dataloader_train = DataLoader(dataset_train, batch_size=batch_size, shuffle=True)
dataset_valid = MyDataset(data_valid)
dataloader_valid = DataLoader(dataset_valid, batch_size=batch_size)
# dataset_test = MyDataset(data_test)
# dataloader_test = DataLoader(dataset_test)
# prepare model
model = MyModel()
model.cuda()
crit = nn.BCEWithLogitsLoss(reduction='none') # reduction is none for OHEM purpose
# optimizer = torch.optim.Adam(model.parameters(), amsgrad=True) # TODO: more settings
# optimizer = torch.optim.Adagrad(model.parameters(), lr=0.01)
# optimizer = torch.optim.Adadelta(model.parameters())
# optimizer = torch.optim.Adamax(model.parameters())
# optimizer = torch.optim.RMSprop(model.parameters())
optimizer = torch.optim.SGD(model.parameters(), lr=0.1)

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


def evaluate(model, phase):

    if phase== 'train':
        dataloader = dataloader_train
    elif phase == 'val':
        dataloader = dataloader_valid
    # elif phase == 'test':
    #     dataloader = dataloader_test
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

    prefix = 'mixed_own_' + phase + '_'
    np.save(file='/opt/hades/' + prefix + 'neg_dists_M_rms_1e2', arr=neg_dists)
    np.save(file='/opt/hades/' + prefix + 'pov_dists_M_rms_1e2', arr=pov_dists)
    np.save(file='/opt/hades/' + prefix + 'neg_dists_rM_rms_1e2', arr=neg_dists_r)
    np.save(file='/opt/hades/' + prefix + 'pov_dists_rM_rms_1e2', arr=pov_dists_r)
    print('rights in {}: {}'.format(phase, count_right))
    print('wrongs in {}: {}'.format(phase, count_wrong))
    return count_wrong / count_right


def train():
    count = -1
    early_stopping = EarlyStopping(patience=15, verbose=True)
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
                    model.load_state_dict(torch.load('/opt/hades/checkpoint.ptM_mixed_own_rms_1e2'))
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