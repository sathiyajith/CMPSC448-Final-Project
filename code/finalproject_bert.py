


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from collections import Counter
from gensim.models import Word2Vec
import transformers
from transformers import BertModel
from transformers import BertTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
from sklearn.metrics import classification_report, confusion_matrix
import time
import random
from data_preprocessing import preprocess
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import RandomOverSampler


def read_data():
    X = df['text_clean'].values
    y = df['sentiment'].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=seed_value)
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, stratify=y_train, random_state=seed_value)
    ros = RandomOverSampler()
    X_train_os, y_train_os = ros.fit_resample(np.array(X_train).reshape(-1,1),np.array(y_train).reshape(-1,1))
    X_train_os = X_train_os.flatten()   
    y_train_os = y_train_os.flatten()
    (unique, counts) = np.unique(y_train_os, return_counts=True)

def bert_tokenizer(data,tokenizer,MAX_LEN):
    input_ids = []
    attention_masks = []
    for sent in data:
        encoded_sent = tokenizer.encode_plus(
            text=sent,
            add_special_tokens=True,
            max_length=MAX_LEN,
            pad_to_max_length=True,
            return_attention_mask=True
            )
        input_ids.append(encoded_sent.get('input_ids'))
        attention_masks.append(encoded_sent.get('attention_mask'))

    input_ids = torch.tensor(input_ids)
    attention_masks = torch.tensor(attention_masks)

    return input_ids, attention_masks


def tokenize(X_train_os,X_valid,X_test,y_train_os,y_valid,y_test):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    encoded_tweets = [tokenizer.encode(sent, add_special_tokens=True) for sent in X_train]
    max_len = max([len(sent) for sent in encoded_tweets])
    print('Max length: ', max_len)
    MAX_LEN = 128
    train_inputs, train_masks = bert_tokenizer(X_train_os)  
    val_inputs, val_masks = bert_tokenizer(X_valid)
    test_inputs, test_masks = bert_tokenizer(X_test)
    train_labels = torch.from_numpy(y_train_os)
    val_labels = torch.from_numpy(y_valid)
    test_labels = torch.from_numpy(y_test)
    batch_size = 32
    train_data = TensorDataset(train_inputs, train_masks, train_labels)
    train_sampler = RandomSampler(train_data)
    train_dataloader = DataLoader(train_data, sampler=train_sampler, batch_size=batch_size)
    val_data = TensorDataset(val_inputs, val_masks, val_labels)
    val_sampler = SequentialSampler(val_data)
    val_dataloader = DataLoader(val_data, sampler=val_sampler, batch_size=batch_size)
    test_data = TensorDataset(test_inputs, test_masks, test_labels)
    test_sampler = SequentialSampler(test_data)
    test_dataloader = DataLoader(test_data, sampler=test_sampler, batch_size=batch_size)

class Bert_Classifier(nn.Module):
    def __init__(self, freeze_bert=False):
        super(Bert_Classifier, self).__init__()
        n_input = 768
        n_hidden = 50
        n_output = 5

        self.bert = BertModel.from_pretrained('bert-base-uncased')

        self.classifier = nn.Sequential(
            nn.Linear(n_input, n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, n_output)
        )
        if freeze_bert:
            for param in self.bert.parameters():
                param.requires_grad = False

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids,
                            attention_mask=attention_mask)
        last_hidden_state_cls = outputs[0][:, 0, :]
        logits = self.classifier(last_hidden_state_cls)

        return logits

def initialize_model(epochs=4,train_dataloader):
    bert_classifier = Bert_Classifier(freeze_bert=False)
    bert_classifier.to(device)
    optimizer = AdamW(bert_classifier.parameters(),lr=5e-5,eps=1e-8)

    total_steps = len(train_dataloader) * epochs

    scheduler = get_linear_schedule_with_warmup(optimizer,
                                                num_warmup_steps=0,
                                                num_training_steps=total_steps)
    return bert_classifier, optimizer, scheduler


def bert_train(model, train_dataloader, val_dataloader=None, epochs=4, evaluation=False):
    print("Start training...\n")
    for epoch_i in range(epochs):
        print("-"*10)
        print("Epoch : {}".format(epoch_i+1))
        print("-"*10)
        print("-"*38)
        print(f"{'BATCH NO.':^7} | {'TRAIN LOSS':^12} | {'ELAPSED (s)':^9}")
        print("-"*38)
        t0_epoch, t0_batch = time.time(), time.time()
        total_loss, batch_loss, batch_counts = 0, 0, 0
        model.train()
        for step, batch in enumerate(train_dataloader):
            batch_counts +=1
            b_input_ids, b_attn_mask, b_labels = tuple(t.to(device) for t in batch)
            model.zero_grad()
            logits = model(b_input_ids, b_attn_mask)
            loss = loss_fn(logits, b_labels)
            batch_loss += loss.item()
            total_loss += loss.item()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            if (step % 100 == 0 and step != 0) or (step == len(train_dataloader) - 1):
                time_elapsed = time.time() - t0_batch
                print(f"{step:^9} | {batch_loss / batch_counts:^12.6f} | {time_elapsed:^9.2f}")
                batch_loss, batch_counts = 0, 0
                t0_batch = time.time()
        avg_train_loss = total_loss / len(train_dataloader)
        model.eval()
        val_accuracy = []
        val_loss = []

        for batch in val_dataloader:
            batch_input_ids, batch_attention_mask, batch_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                logits = model(batch_input_ids, batch_attention_mask)
            loss = loss_fn(logits, batch_labels)
            val_loss.append(loss.item())
            preds = torch.argmax(logits, dim=1).flatten()
            accuracy = (preds == batch_labels).cpu().numpy().mean() * 100
            val_accuracy.append(accuracy)
        val_loss = np.mean(val_loss)
        val_accuracy = np.mean(val_accuracy)
        time_elapsed = time.time() - t0_epoch
        print(f"{'AVG TRAIN LOSS':^12} | {'VAL ACCURACY (%)':^9}")
        print(f"{avg_train_loss:^14.6f} | {val_accuracy:^17.2f}")
        print("\n")

    print("Training complete!")

def bert_predict(model, test_dataloader):
    preds_list = []
    model.eval()

    for batch in test_dataloader:
        batch_input_ids, batch_attention_mask = tuple(t.to(device) for t in batch)[:2]
        with torch.no_grad():
            logit = model(batch_input_ids, batch_attention_mask)
        pred = torch.argmax(logit,dim=1).cpu().numpy()
        preds_list.extend(pred)

    return preds_list


if __name__ =="__main__":
    df = preprocess()
    df,max_len, X_train,X_train_os,X_valid,X_test,y_train_os,y_valid,y_test  = read_data(df)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_dataloader, val_dataloader = tokenize(X_train_os,X_valid,X_test,y_train_os,y_valid,y_test)
    bert_classifier, optimizer, scheduler = initialize_model(epochs=2)
    loss_fn = nn.CrossEntropyLoss()
    test_dataloader, y_test = bert_train(bert_classifier, train_dataloader, val_dataloader, epochs=2)
    bert_preds = bert_predict(bert_classifier, test_dataloader)
    print('Classification Report for BERT :\n', classification_report(y_test, bert_preds))

