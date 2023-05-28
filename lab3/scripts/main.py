import os
import warnings

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader

from config import EMB_PATH
from dataloading import SentenceDataset
from models import BaselineDNN
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from nltk.tokenize import word_tokenize
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report

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
BATCH_SIZE = 128
EPOCHS = 50
DATASET = "Semeval2017A"  # options: "MR", "Semeval2017A"

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

# Define our PyTorch-based Dataset
'''lengths = [len(word_tokenize(sent)) for sent in X_train]
df = pd.DataFrame(lengths)
ax = df.hist(bins = max(lengths), figsize=(12,8), color='#86bf91', zorder=2, rwidth=0.9)

for x in ax[0]:
  x.set_title("Histogram for the length of the sentences", weight='bold', size=12)
  x.set_xlabel("Length", labelpad=20, weight='bold', size=12)
  x.set_ylabel("Number of sentences", labelpad=20, weight='bold', size=12)
plt.show()'''


train_set = SentenceDataset(X_train, y_train, word2idx)
test_set = SentenceDataset(X_test, y_test, word2idx)

# EX7 - Define our PyTorch-based DataLoader
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True) # EX7
test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7

'''for i, j, k in train_loader:
    print(i)
    print(len(i))
    for l in i:
        print(l.shape)
    print()
    print(j)
    print()
    print(k)
    break'''

#############################################################################
# Model Definition (Model, Loss Function, Optimizer)
#############################################################################
output_size = n_classes if n_classes > 2 else 1
model = BaselineDNN(output_size=output_size,  # EX8
                    embeddings=embeddings,
                    trainable_emb=EMB_TRAINABLE)

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
test_loss_per_epoch = []
for epoch in range(1, EPOCHS + 1):
    # train the model for one epoch
    train_dataset(epoch, train_loader, model, criterion, optimizer, n_classes)

    # evaluate the performance of the model, on both data sets
    train_loss, (y_train_pred, y_train_gold) = eval_dataset(train_loader,
                                                            model,
                                                            criterion,
                                                            n_classes)

    train_loss_per_epoch.append(train_loss)

    test_loss, (y_test_pred, y_test_gold) = eval_dataset(test_loader,
                                                         model,
                                                         criterion,
                                                         n_classes)

    test_loss_per_epoch.append(test_loss)
    
    #print('Epoch [{}/{}], Train loss: {:.4f}, Test loss: {:.4f}'.format(epoch, EPOCHS, train_loss, test_loss))

print('\nClassification report for the test data:\n')
print(classification_report(y_test_gold, y_test_pred))

plt.plot(train_loss_per_epoch)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss')
plt.show()

plt.plot(test_loss_per_epoch)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Test Loss')
plt.show()