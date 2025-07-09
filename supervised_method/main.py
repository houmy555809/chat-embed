import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import math
import copy
import jieba
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.cluster import *
import pandas as pd
import sys

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
cmap = plt.get_cmap("viridis")

word_list = ["<start>", "<end>"]

df = pd.read_csv("sentences.csv")
sentences = df["text"]
raw_sentences = copy.deepcopy(sentences)
category_list = df["category"]
categories = {}
for category in category_list:
    if category not in categories:
        categories[category] = len(categories)

def preprocess_sentence(sentence):
    arr = jieba.lcut(sentence)
    arr = ["<start>"] + arr + ["<end>"]
    return arr

sentences = [preprocess_sentence(sentence) for sentence in sentences]

vocab = {}
def build_vocab():
    global vocab
    for sentence in sentences:
        for word in sentence:
            if word not in vocab.keys():
                _id = len(vocab)
                vocab[word] = _id

build_vocab()
converted_sentences = [
    [vocab[word] for word in sentence] for sentence in sentences
]
rev_vocab = list(vocab.keys())

class Encoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, 32)
        self.lstm = nn.LSTM(32, 16, bidirectional = True)
        self.nn = nn.Linear(32, 32)
    def forward(self, x):
        x = self.embed(x)
        lstm_out, _ = self.lstm(x)
        return self.nn(lstm_out[-1].view(-1))

vocab_size = len(vocab)
encoder = Encoder(vocab_size)

class Classifier(nn.Module):
    def __init__(self, vocab_size, num_categories):
        super().__init__()
        self.nn = nn.Sequential(
            nn.Linear(32, 16),
            nn.Sigmoid(),
            nn.Linear(16, num_categories)
        )
    
    def forward(self, inp):
        return self.nn(inp)

classifier = Classifier(vocab_size, len(categories))

def progress(percentage):
    percentage = int(percentage)
    return '#' * percentage + '-' * (100 - percentage)

def train(epoch_id = 1):
    encoder.train()
    classifier.train()
    optim_enc = optim.Adam(encoder.parameters())
    optim_dec = optim.Adam(classifier.parameters())
    criterion = nn.CrossEntropyLoss()
    tot_loss = 0
    acc = 0
    for i, sentence in enumerate(converted_sentences):
        optim_enc.zero_grad()
        optim_dec.zero_grad()
        encoded = encoder(torch.tensor(sentence, dtype = torch.long))
        pred = classifier(encoded)
        category = categories[category_list[i]]
        loss = criterion(pred, torch.tensor(category, dtype = torch.long))
        loss.backward()
        optim_enc.step()
        optim_dec.step()
        tot_loss += loss.item()
        acc += torch.argmax(pred).item() == category
    print("Epoch %d loss = %.2f acc = %.2f%%" % (epoch_id, tot_loss, acc / len(converted_sentences) * 100))
    return tot_loss, acc / len(converted_sentences)

def save():
    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(classifier.state_dict(), "classifier.pth")

def load():
    global encoder, decoder
    encoder.load_state_dict(torch.load("encoder.pth"))
    classifier.load_state_dict(torch.load("classifier.pth"))

def get_embed(sentence):
    raw = preprocess_sentence(sentence)
    converted = [vocab[word] for word in raw]
    return encoder(torch.tensor(converted, dtype = torch.long))

def plot_embed(sentences):
    pca = PCA(n_components = 3)
    data = [get_embed(sentence).detach().numpy() for sentence in sentences]
    embedded = pca.fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    for i in range(len(sentences)):
        ax.plot([0, embedded[i][0]], [0, embedded[i][1]], [0, embedded[i][2]], color = cmap(categories[category_list[i]] / len(categories)))
    for i in range(len(categories)):
        ax.plot([0], [0], [0], color = cmap(i / len(categories)), label = list(categories.keys())[i])
    ax.legend()
    ax.axis()
    plt.show()

def get_embed_dict(sentences):
    res = {}
    for sentence in sentences:
        res[sentence] = get_embed(sentence).detach().numpy()
    return res

def get_pca_embed_dict(sentences):
    pca = PCA(n_components = 3)
    data = [get_embed(sentence).detach().numpy() for sentence in sentences]
    embedded = pca.fit_transform(data)
    res = {}
    for i in range(len(sentences)):
        res[sentences[i]] = (embedded[i][0], embedded[i][1], embedded[i][2])
    return res

def cluster(sentences, num = 5):
    embed_dict = get_embed_dict(sentences)
    algo = SpectralClustering(num)
    algo = algo.fit(list(embed_dict.values()))
    for i in range(len(embed_dict)):
        print(list(embed_dict.keys())[i], "\t ==>", algo.labels_[i])
    pca = PCA(n_components = 3)
    data = [get_embed(sentence).detach().numpy() for sentence in list(embed_dict.keys())]
    embedded = pca.fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    for i in range(len(embed_dict)):
        ax.plot([0, embedded[i][0]], [0, embedded[i][1]], [0, embedded[i][2]], color = cmap(algo.labels_[i] / num))
    for i in range(num):
        ax.plot([0], [0], [0], color = cmap(i / num), label = str(i))
    ax.legend()
    ax.axis()
    plt.show()

load()
# uncomment this to train
"""
loss_history = []
acc_history = []

for i in range(1, 31):
    loss, acc = train(i)
    loss_history.append(loss)
    acc_history.append(acc)
    save()
"""
