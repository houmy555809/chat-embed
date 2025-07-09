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
import sys

plt.rcParams["font.sans-serif"] = ["SimHei"]
plt.rcParams["axes.unicode_minus"] = False
cmap = plt.get_cmap("viridis")

word_list = ["<start>", "<end>"]

sentences = []
with open("sentences.txt", encoding = "utf-8") as file:
    sentences = [sentence.strip() for sentence in file.readlines()]

raw_sentences = copy.deepcopy(sentences)

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
        self.embed = nn.Embedding(vocab_size, 128)
        self.lstm = nn.LSTM(128, 64, bidirectional = True)
    def forward(self, x):
        x = self.embed(x)
        lstm_out, _ = self.lstm(x)
        return lstm_out[-1].view(-1)

vocab_size = len(vocab)
encoder = Encoder(vocab_size)

class Decoder(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.cell = nn.LSTMCell(256, 128)
        self.embed = encoder.embed

    def decode_embed(self, vec):
        cos_similarities = nn.functional.cosine_similarity(self.embed.weight, vec, dim = 1)
        _, predicted_index = cos_similarities.max(0)
        return predicted_index
    
    def forward(self, inp):
        h = torch.zeros(128)
        C = torch.zeros(128)
        x = self.embed(torch.tensor([vocab["<start>"]])).view(-1)
        res = []
        decoded = []
        _len = 0
        while True:
            h, C = self.cell(torch.cat((x, inp)).view(-1), (h, C))
            res.append(C)
            dec = self.decode_embed(C).detach().item()
            decoded.append(dec)
            x = C
            _len += 1
            if rev_vocab[dec] == "<end>" or _len >= 64: break
        return (res, decoded)

decoder = Decoder(vocab_size)

def progress(percentage):
    percentage = int(percentage)
    return '#' * percentage + '-' * (100 - percentage)

def train(epoch_id = 1):
    encoder.train()
    decoder.train()
    optim_enc = optim.Adam(encoder.parameters())
    optim_dec = optim.Adam(decoder.parameters())
    criterion = nn.MSELoss()
    for i, sentence in enumerate(converted_sentences):
        optim_enc.zero_grad()
        optim_dec.zero_grad()
        encoded = encoder(torch.tensor(sentence, dtype = torch.long))
        decoded, recovered = decoder(encoded)
        orig_sentence = ''.join([rev_vocab[word] for word in sentence])
        recovered_sentence = ''.join([rev_vocab[word] for word in recovered])
        sys.stdout.write('\r' + ' ' * 200 + \
                         '\r' + orig_sentence + " ===> " + recovered_sentence + '\n' + \
                         "%04d %3d/%3d %3d%% %s" % (epoch_id, i + 1, len(sentences), int((i + 1) / len(sentences) * 100), progress((i + 1) / len(sentences) * 100)))
        loss = 0
        for i in range(min(len(sentence) - 1, len(decoded))):
            tgt = encoder.embed(torch.tensor([sentence[i + 1]], dtype = torch.long)).detach().view(-1)
            got = decoded[i].view(-1)
            if recovered[i] != vocab["<end>"] and recovered[i] != vocab["<start>"]: loss += criterion(got, tgt)
            else: loss += 10 * criterion(got, tgt)
        loss.backward()
        optim_enc.step()
        optim_dec.step()

def save():
    torch.save(encoder.state_dict(), "encoder.pth")
    torch.save(decoder.state_dict(), "decoder.pth")

def load():
    global encoder, decoder
    encoder.load_state_dict(torch.load("encoder.pth"))
    decoder.load_state_dict(torch.load("decoder.pth"))

def get_embed(sentence):
    raw = preprocess_sentence(sentence)
    converted = [vocab[word] for word in raw]
    return encoder(torch.tensor(converted, dtype = torch.long))

def get_decoded(embed):
    decoded, recovered = decoder(encoded)
    return [rev_vocab[word] for word in sentence]

def plot_embed(sentences):
    pca = PCA(n_components = 3)
    data = [get_embed(sentence).detach().numpy() for sentence in sentences]
    embedded = pca.fit_transform(data)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection = "3d")
    for i in range(len(sentences)):
        ax.plot([0, embedded[i][0]], [0, embedded[i][1]], [0, embedded[i][2]], label = sentences[i], color = cmap(i / len(sentences)))
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

# uncomment this to train
"""
load()

for i in range(1, 501):
    train(i)
    save()
"""
