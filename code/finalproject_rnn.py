import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from collections import Counter
from gensim.models import Word2Vec
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
import time
import random
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler
from data_preprocessing import preprocess


def read_data(df):
    X = df['text_clean']
    y = df['sentiment']
    ros = RandomOverSampler()
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed_value)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed_value)
    X_train, y_train = ros.fit_resample(np.array(X_train).reshape(-1, 1), np.array(y_train).reshape(-1, 1));
    train_os = pd.DataFrame(list(zip([x[0] for x in X_train], y_train)), columns = ['text_clean', 'sentiment']);
    X_train = train_os['text_clean'].values
    y_train = train_os['sentiment'].values
    (unique, counts) = np.unique(y_train, return_counts=True)
    np.asarray((unique, counts)).T

def Tokenize(column, seq_len):
    corpus = [word for text in column for word in text.split()]
    count_words = Counter(corpus)
    sorted_words = count_words.most_common()
    vocab_to_int = {w:i+1 for i, (w,c) in enumerate(sorted_words)}

    text_int = []
    for text in column:
        r = [vocab_to_int[word] for word in text.split()]
        text_int.append(r)

    features = np.zeros((len(text_int), seq_len), dtype = int)
    for i, review in enumerate(text_int):
        if len(review) <= seq_len:
            zeros = list(np.zeros(seq_len - len(review)))
            new = zeros + review
        else:
            new = review[: seq_len]
        features[i, :] = np.array(new)

    return sorted_words, features

def get_embedding(df,max_len, X_train):
    vocabulary, tokenized_column = Tokenize(df["text_clean"], max_len)
    df["text_clean"].iloc[0]
    tokenized_column[10]
    keys = []
    values = []
    for key, value in vocabulary[:20]:
        keys.append(key)
        values.append(value)
    Word2vec_train_data = list(map(lambda x: x.split(), X_train))
    EMBEDDING_DIM = 200
    word2vec_model = Word2Vec(Word2vec_train_data, vector_size=EMBEDDING_DIM)
    print(f"Vocabulary size: {len(vocabulary) + 1}")
    VOCAB_SIZE = len(vocabulary) + 1
    embedding_matrix = np.zeros((VOCAB_SIZE, EMBEDDING_DIM))
    for word, token in vocabulary:
        if word in word2vec_model.wv.key_to_index:
            embedding_vector = word2vec_model.wv[word]
            embedding_matrix[token] = embedding_vector
    print("Embedding Matrix Shape:", embedding_matrix.shape)


def split_data(tokenized_column):
    X = tokenized_column
    y = df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed_value)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed_value)
    (unique, counts) = np.unique(y_train, return_counts=True)
    ros = RandomOverSampler()
    X_train_os, y_train_os = ros.fit_resample(np.array(X_train),np.array(y_train));
    (unique, counts) = np.unique(y_train_os, return_counts=True)
    train_data = TensorDataset(torch.from_numpy(X_train_os), torch.from_numpy(y_train_os))  
    test_data = TensorDataset(torch.from_numpy(X_test), torch.from_numpy(y_test))
    valid_data = TensorDataset(torch.from_numpy(X_valid), torch.from_numpy(y_valid))
    BATCH_SIZE = 32
    train_loader = DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE, drop_last=True)
    valid_loader = DataLoader(valid_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)
    test_loader = DataLoader(test_data, shuffle=False, batch_size=BATCH_SIZE, drop_last=True)

class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attn = nn.Linear(hidden_dim * 2, hidden_dim)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, hidden, encoder_outputs):
        seq_len = encoder_outputs.size(1)
        hidden = hidden[-1]
        hidden_repeated = hidden.unsqueeze(1).repeat(1, seq_len, 1)
        attn_weights = torch.tanh(self.attn(torch.cat((hidden_repeated, encoder_outputs), dim=2)))
        attn_weights = self.v(attn_weights).squeeze(2)
        return nn.functional.softmax(attn_weights, dim=1)


class RNN_Sentiment_Classifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, num_classes, rnn_layers, dropout):
        super(RNN_Sentiment_Classifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = rnn_layers
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.rnn = nn.LSTM(embedding_dim, hidden_dim, rnn_layers, batch_first=True)
        self.attention = Attention(hidden_dim)
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.softmax = nn.LogSoftmax(dim=1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, hidden):
        embedded = self.embedding(x)
        out, hidden = self.rnn(embedded, hidden)
        attn_weights = self.attention(hidden[0], out)
        context = attn_weights.unsqueeze(1).bmm(out).squeeze(1)
        out = self.softmax(self.fc(context))
        return out, hidden

    def init_hidden(self, batch_size):
        factor = 1
        h0 = torch.zeros(self.num_layers * factor, batch_size, self.hidden_dim).to(DEVICE)
        c0 = torch.zeros(self.num_layers * factor, batch_size, self.hidden_dim).to(DEVICE)
        hidden = torch.zeros(self.num_layers * factor, batch_size, self.hidden_dim).to(DEVICE)



NUM_CLASSES = 5
HIDDEN_DIM = 100
RNN_LAYERS = 1
LR = 4e-4
DROPOUT = 0.5
EPOCHS = 10
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def train(train_loader,valid_loader,embedding_matrix,VOCAB_SIZE,EMBEDDING_DIM):
    model = RNN_Sentiment_Classifier(VOCAB_SIZE, EMBEDDING_DIM, HIDDEN_DIM, NUM_CLASSES, RNN_LAYERS, DROPOUT)
    model = model.to(DEVICE)
    model.embedding.weight.data.copy_(torch.from_numpy(embedding_matrix))
    model.embedding.weight.requires_grad = True
    criterion = nn.NLLLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay = 5e-6)
    print(model)
    total_step = len(train_loader)
    total_step_val = len(valid_loader)
    early_stopping_patience = 4
    early_stopping_counter = 0
    valid_acc_max = 0

    for e in range(EPOCHS):
        train_loss, valid_loss  = [], []
        train_acc, valid_acc  = [], []
        y_train_list, y_val_list = [], []
        correct, correct_val = 0, 0
        total, total_val = 0, 0
        running_loss, running_loss_val = 0, 0
        model.train()
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            h = model.init_hidden(labels.size(0))
            model.zero_grad()
            output, h = model(inputs,h)
            loss = criterion(output, labels)
            loss.backward()
            running_loss += loss.item()
            optimizer.step()
            y_pred_train = torch.argmax(output, dim=1)
            y_train_list.extend(y_pred_train.squeeze().tolist())
            correct += torch.sum(y_pred_train==labels).item()
            total += labels.size(0)

        train_loss.append(running_loss / total_step)
        train_acc.append(100 * correct / total)
        with torch.no_grad():
            model.eval()
            for inputs, labels in valid_loader:
                inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
                val_h = model.init_hidden(labels.size(0))
                output, val_h = model(inputs, val_h)
                val_loss = criterion(output, labels)
                running_loss_val += val_loss.item()
                y_pred_val = torch.argmax(output, dim=1)
                y_val_list.extend(y_pred_val.squeeze().tolist())
                correct_val += torch.sum(y_pred_val==labels).item()
                total_val += labels.size(0)
            valid_loss.append(running_loss_val / total_step_val)
            valid_acc.append(100 * correct_val / total_val)
        if np.mean(valid_acc) >= valid_acc_max:
            torch.save(model.state_dict(), './state_dict.pt')
            print(f'Epoch {e+1}:Validation accuracy increased ({valid_acc_max:.6f} --> {np.mean(valid_acc):.6f}).  Saving model ...')
            valid_acc_max = np.mean(valid_acc)
            early_stopping_counter=0
        else:
            print(f'Epoch {e+1}:Validation accuracy did not increase')
            early_stopping_counter+=1
        if early_stopping_counter > early_stopping_patience:
            print('Early stopped at epoch :', e+1)
            break
        print(f'\tTrain_loss : {np.mean(train_loss):.4f} Val_loss : {np.mean(valid_loss):.4f}')
        print(f'\tTrain_acc : {np.mean(train_acc):.3f}% Val_acc : {np.mean(valid_acc):.3f}%')
    model.load_state_dict(torch.load('./state_dict.pt'))
    return model, test_loader

def evaluate_model(model, test_loader):
    model.eval()
    y_pred_list = []
    y_test_list = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(DEVICE), labels.to(DEVICE)
            test_h = model.init_hidden(labels.size(0))

            output, val_h = model(inputs, test_h)
            y_pred_test = torch.argmax(output, dim=1)
            y_pred_list.extend(y_pred_test.squeeze().tolist())
            y_test_list.extend(labels.squeeze().tolist())

    return y_pred_list, y_test_list


if __name__ =="__main__":
    df = preprocess()
    df,max_len, X_train = read_data(df)
    tokenized_column = get_embedding(df,max_len, X_train)
    train_loader,valid_loader,embedding_matrix,VOCAB_SIZE,EMBEDDING_DIM = split_data(tokenized_column)
    model,test_loader = train(train_loader,valid_loader,embedding_matrix,VOCAB_SIZE,EMBEDDING_DIM)
    y_pred_list, y_test_list = evaluate_model(model, test_loader)
    print('RNN Classification Result :\n', classification_report(y_test_list, y_pred_list))
