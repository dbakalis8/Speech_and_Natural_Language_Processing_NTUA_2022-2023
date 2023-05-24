# Train a torch DNN for Kaldi DNN-HMM model
import math
import sys

import numpy as np
import torch
import torch.utils.data
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from dnn.torch_dataset import TorchSpeechDataset
from dnn.torch_dnn import TorchDNN

DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# DEVICE = torch.device('cpu')
# CONFIGURATION #

NUM_LAYERS = 2
HIDDEN_DIM = [128]
USE_BATCH_NORM = True
DROPOUT_P = .2
EPOCHS = 50
PATIENCE = 3

if len(sys.argv) < 2:
    print("USAGE: python timit_dnn.py <PATH/TO/CHECKPOINT_TO_SAVE.pt>")

BEST_CHECKPOINT = sys.argv[1]


# FIXME: You may need to change these paths
TRAIN_ALIGNMENT_DIR = "exp/tri_align_train"
DEV_ALIGNMENT_DIR = "exp/tri_align_dev"
TEST_ALIGNMENT_DIR = "exp/tri_align_test"


def train(model, criterion, optimizer, train_loader, dev_loader, epochs=50, patience=3):
    """Train model using Early Stopping and save the checkpoint for
    the best validation loss
    """
    # TODO: IMPLEMENT THIS FUNCTION
    best_dev_loss = 1e9
    model.train()
    print(len(train_loader), len(dev_loader))
    cnt = 0
    for epoch in range(epochs):
        print(epoch)
        total_train_loss = 0.0
        cnt1 = 1
        for features, labels in train_loader:
            if cnt1 % 500 == 0:
                print(cnt1)
            #print(labels.shape)
            #print(labels)
            #print(features.shape)
            preds = model(features)
            #print(preds.shape)
            #print(preds)
            #gjh
            train_loss = criterion(preds, labels)
            total_train_loss += train_loss

            optimizer.zero_grad()
            train_loss.backward()
            optimizer.step()
            cnt1 += 1
            if cnt1 == 4000:
                break

        model.eval()
        dev_loss = 0.0
        cnt2 = 1
        with torch.no_grad():
            for features, labels in dev_loader:
                if cnt2 % 100 == 0:
                    print(cnt2)
                preds = model(features)
                dev_loss += criterion(preds, labels)
                cnt2 += 1

        if dev_loss < best_dev_loss:
            best_dev_loss = dev_loss
            torch.save(model, BEST_CHECKPOINT)
            cnt = 0
        else:
            cnt += 1
            if cnt == patience:
                print('Early stopping ...')
                break

        print('Epoch {}, Train loss: {}, Dev loss: {}'.format(epoch, total_train_loss.item()/cnt1,  dev_loss.item()/cnt2))
        #break

#print(0)
trainset = TorchSpeechDataset('./', TRAIN_ALIGNMENT_DIR, 'train')
validset = TorchSpeechDataset('./', DEV_ALIGNMENT_DIR, 'dev')
testset = TorchSpeechDataset('./', TEST_ALIGNMENT_DIR, 'test')
#print(1)
scaler = StandardScaler()
scaler.fit(trainset.feats)
#print(2)

trainset.feats = scaler.transform(trainset.feats)
validset.feats = scaler.transform(validset.feats)
testset.feats = scaler.transform(testset.feats)
#print(3)

feature_dim = trainset.feats.shape[1]
n_classes = int(trainset.labels.max() - trainset.labels.min() + 1)
#print(feature_dim, n_classes)

#print(4)
dnn = TorchDNN(
    feature_dim,
    n_classes,
    num_layers=NUM_LAYERS,
    batch_norm=USE_BATCH_NORM,
    hidden_dim=HIDDEN_DIM,
    dropout_p=DROPOUT_P
)
dnn.to(DEVICE)
#print(dnn)
#print(5)
train_loader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True)
dev_loader = torch.utils.data.DataLoader(validset, batch_size=128, shuffle=True)

optimizer = torch.optim.Adam(dnn.parameters())
criterion = torch.nn.CrossEntropyLoss()
#print(6)
# for i in train_loader:
#     print(i)

train(dnn, criterion, optimizer, train_loader, dev_loader, epochs=EPOCHS, patience=PATIENCE)
