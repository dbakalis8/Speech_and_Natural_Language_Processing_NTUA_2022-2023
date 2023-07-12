import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN, LSTM
from training import train_dataset, eval_dataset, torch_train_val_split, get_metrics_report
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report
from early_stopper import EarlyStopper
from attention import SimpleSelfAttentionModel, MultiHeadAttentionModel, TransformerEncoderModel

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")

# 2 - set the correct dimensionality of the embeddings
EMB_DIM = 50

EMB_TRAINABLE = False
BIDIRECTIONAL_LSTM = True
BATCH_SIZE = 128
EPOCHS = 20
DATASET = "Semeval2017A" # options: "MR", "Semeval2017A"
MAX_LENGTH = 40

# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# load the raw data
if DATASET == "Semeval2017A":
    X_train, y_train, X_test, y_test = load_Semeval2017A()
elif DATASET == "MR":
    X_train, y_train, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")

# convert data labels from strings to integers
le = LabelEncoder()
le.fit(y_train)
y_train = le.transform(y_train)  # EX1
y_test = le.transform(y_test)  # EX1
n_classes = le.classes_.size  # EX1 - LabelEncoder.classes_.size

# for i in range(10):
#     print(X_train[i], y_train[i])

# Define our PyTorch-based Dataset
# lengths = [len(word_tokenize(sent)) for sent in X_train]
# df = pd.DataFrame(lengths)
# ax = df.hist(bins = max(lengths), figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)

# for x in ax[0]:
#   x.set_title("Histogram for the length of the sentences", weight='bold', size=12)
#   x.set_xlabel("Length", labelpad=20, weight='bold', size=12)
#   x.set_ylabel("Number of sentences", labelpad=20, weight='bold', size=12)
# plt.show()


train_set = SentenceDataset(X_train, y_train, word2idx, MAX_LENGTH)
test_set = SentenceDataset(X_test, y_test, word2idx, MAX_LENGTH)

# for i in range(5):
#     print(train_set[i])

# EX7 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
output_size = n_classes

baseline_dnn = BaselineDNN(output_size=output_size,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE,
                    max_pooling=False)

lstm = LSTM(output_size=output_size, 
            embeddings=embeddings, 
            trainable_emb=EMB_TRAINABLE, 
            bidirectional=BIDIRECTIONAL_LSTM)

sa = SimpleSelfAttentionModel(output_size=output_size, 
                              embeddings=embeddings, 
                              max_length=MAX_LENGTH)

mha = MultiHeadAttentionModel(output_size=output_size, 
                              embeddings=embeddings, 
                              max_length=MAX_LENGTH, 
                              n_head=5)

transformer = TransformerEncoderModel(output_size=output_size, 
                                      embeddings=embeddings, 
                                      max_length=MAX_LENGTH, 
                                      n_head=5, 
                                      n_layer=3)

model = baseline_dnn

# move the mode weight to cpu or gpu
model.to(DEVICE)
print(model)

# We optimize ONLY those parameters that are trainable (p.requires_grad==True)
criterion = torch.nn.BCEWithLogitsLoss() if n_classes == 2 else torch.nn.CrossEntropyLoss()  # EX8
parameters = filter(lambda p: p.requires_grad, model.parameters()) # EX8
optimizer = torch.optim.Adam(parameters)  # EX8

#############################################################################
# Training Pipeline
#############################################################################

train_loss_per_epoch = []
val_loss_per_epoch = []

train_loader, val_loader = torch_train_val_split(train_set, BATCH_SIZE, BATCH_SIZE)
save_path = f'./trained_models/best_{model.__class__.__name__}_{DATASET}_embdim{EMB_DIM}.pt'
early_stop = EarlyStopper(model, save_path, patience=5)

for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer, n_classes)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_pred, y_train_gold) = eval_dataset(train_loader,
                                                            model,
                                                            criterion,
                                                            n_classes)

    train_loss_per_epoch.append(train_loss)

    val_loss, (y_val_pred, y_val_gold) = eval_dataset(val_loader,
                                                         model,
                                                         criterion,
                                                         n_classes)

    val_loss_per_epoch.append(val_loss)

    print('Epoch [{}/{}], Train loss: {:.4f}, Val loss: {:.4f}'.format(epoch, EPOCHS, train_loss, val_loss))

    if early_stop.early_stop(val_loss):
        print('early stopping ...')
        break

model.load_state_dict(torch.load(save_path))

_, (y_test_pred, y_test_gold) = eval_dataset(test_loader,
                                             model,
                                             criterion,
                                             n_classes)

print('\nMetrics report for the test data:\n')
print(get_metrics_report([np.array(y_test_gold)], [np.array(y_test_pred)]))

plt.plot(train_loss_per_epoch)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

plt.plot(val_loss_per_epoch)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Validation Loss')
plt.show()