from tqdm import tqdm
import matplotlib.pyplot as plt
import pandas as pd
import torch
from torchtext.data import Field, TabularDataset, BucketIterator, Iterator
import torch.nn as nn
from transformers import BertTokenizer, BertForSequenceClassification
import torch.optim as optim

from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import seaborn as sns


def buildDataSets(batch_size=8):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    # Model parameter
    MAX_SEQ_LEN = 16
    PAD_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.pad_token)
    UNK_INDEX = tokenizer.convert_tokens_to_ids(tokenizer.unk_token)

    # Fields

    label_field = Field(sequential=False, use_vocab=False, batch_first=True, dtype=torch.int8)
    text_field = Field(use_vocab=False, tokenize=tokenizer.encode, lower=False, include_lengths=False, batch_first=True,
                    fix_length=MAX_SEQ_LEN, pad_token=PAD_INDEX, unk_token=UNK_INDEX)

    fields = {'label': ('label', label_field), 'text': ('text', text_field)}

    # TabularDataset

    train, valid, test = TabularDataset.splits(path='memesData/data', train='train.jsonl', validation='dev_unseen.jsonl',
                                            test='dev_seen.jsonl', format='JSON', fields=fields)

    # Iterators

    train_iter = BucketIterator(train,
                                batch_size=batch_size,
                                sort_key=lambda x: len(x.text),
                                device=device, train=True,
                                sort=True,
                                sort_within_batch=True)
    valid_iter = BucketIterator(valid,
                                batch_size=batch_size,
                                sort_key=lambda x: len(x.text),
                                device=device, train=True,
                                sort=True,
                                sort_within_batch=True)
    test_iter = Iterator(test,
                         batch_size=batch_size,
                         device=device,
                         train=False,
                         shuffle=False,
                         sort=False)

    return train_iter, valid_iter, test_iter


class BERT(nn.Module):

    def __init__(self):
        super(BERT, self).__init__()

        options_name = "bert-base-uncased"
        self.encoder = BertForSequenceClassification.from_pretrained(options_name, num_labels = 2)

    def forward(self, text, label):
        loss, text_fea = self.encoder(text, labels=label)[:2]
        return loss, text_fea


def save_checkpoint(save_path, model, valid_loss):
    state_dict = {'model_state_dict': model.state_dict(),
                  'valid_loss': valid_loss}

    torch.save(state_dict, save_path)
    print(f'Model saved to: {save_path}')

def load_checkpoint(load_path, model):
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    model.load_state_dict(state_dict['model_state_dict'])
    return state_dict['valid_loss']


def save_metrics(save_path, train_loss_list, valid_loss_list, global_steps_list):
    state_dict = {'train_loss_list': train_loss_list,
                  'valid_loss_list': valid_loss_list,
                  'global_steps_list': global_steps_list}

    torch.save(state_dict, save_path)
    print(f'Model saved to ==> {save_path}')


def load_metrics(load_path):
    state_dict = torch.load(load_path, map_location=device)
    print(f'Model loaded from <== {load_path}')

    return state_dict['train_loss_list'], state_dict['valid_loss_list'], state_dict['global_steps_list']

def train(model,
          optimizer,
          train_loader,
          valid_loader,
          verbose=True):
    num_epochs = 25
    eval_every = len(train_loader) // 2
    best_valid_loss = float("Inf")
    criterion = nn.BCELoss()

    # initialize running values
    running_loss = 0.0
    valid_running_loss = 0.0
    global_step = 0
    train_loss_list = []
    valid_loss_list = []
    global_steps_list = []

    # training loop
    model.train()
    for epoch in range(num_epochs):
        if(verbose):
            print("Epoch %d" % epoch)
        with tqdm(train_loader, unit="batch") as tepoch:
            for (labels, text), _ in tepoch:
                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                text = text.type(torch.LongTensor)
                text = text.to(device)
                output = model(text, labels)
                loss, _ = output

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # update running values
                running_loss += loss.item()
                global_step += 1

                # evaluation step
                if global_step % eval_every == 0:
                    model.eval()
                    with torch.no_grad():

                        # validation loop
                        for (labels, text), _ in valid_loader:
                            labels = labels.type(torch.LongTensor)
                            labels = labels.to(device)
                            text = text.type(torch.LongTensor)
                            text = text.to(device)
                            output = model(text, labels)
                            loss, _ = output

                            valid_running_loss += loss.item()

                    # evaluation
                    average_train_loss = running_loss / eval_every
                    average_valid_loss = valid_running_loss / len(valid_loader)
                    train_loss_list.append(average_train_loss)
                    valid_loss_list.append(average_valid_loss)
                    global_steps_list.append(global_step)

                    # resetting running values
                    running_loss = 0.0
                    valid_running_loss = 0.0
                    model.train()

                    # print progress
                    print('Epoch [{}/{}], Step [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'
                          .format(epoch+1, num_epochs, global_step, num_epochs*len(train_loader),
                                  average_train_loss, average_valid_loss))

                    # checkpoint
                    if best_valid_loss > average_valid_loss:
                        best_valid_loss = average_valid_loss
                        save_checkpoint(modelPath, model, best_valid_loss)
                        save_metrics(metricsPath, train_loss_list, valid_loss_list, global_steps_list)

    save_metrics(metricsPath, train_loss_list, valid_loss_list, global_steps_list)
    print('Finished Training!')


def evaluate(model, test_loader):
    y_pred = []
    y_true = []

    model.eval()
    with torch.no_grad():
        for (labels, text), _ in test_loader:

                labels = labels.type(torch.LongTensor)
                labels = labels.to(device)
                text = text.type(torch.LongTensor)
                text = text.to(device)
                output = model(text, labels)

                _, output = output
                y_pred.extend(torch.argmax(output, 1).tolist())
                y_true.extend(labels.tolist())

    print('Classification Report:')
    print(classification_report(y_true, y_pred, labels=[0,1], digits=4))

    cm = confusion_matrix(y_true, y_pred, labels=[0,1])
    ax= plt.subplot()
    sns.heatmap(cm, annot=True, ax = ax, cmap='Blues', fmt="d")

    ax.set_title('Confusion Matrix')

    ax.set_xlabel('Predicted Labels')
    ax.set_ylabel('True Labels')

    ax.xaxis.set_ticklabels(['0', '1'])
    ax.yaxis.set_ticklabels(['0', '1'])
    plt.savefig('eval.png')

if __name__ == "__main__":
    device = torch.device("cuda:0")


    modelDataFolder = 'bertData'
    modelPath = modelDataFolder + '/model.pt'
    metricsPath = modelDataFolder + '/metrics.pt'
    print("Creating model")
    model = BERT().to(device)

    optimizer = optim.Adam(model.parameters(), lr=1e-5)
    print("loading data")
    trainSet, validSet, testSet = buildDataSets()
    print("training")
    train(model, optimizer, trainSet, validSet)
    train_loss_list, valid_loss_list, global_steps_list = load_metrics(metricsPath)
    plt.plot(global_steps_list, train_loss_list, label='Train')
    plt.plot(global_steps_list, valid_loss_list, label='Valid')
    plt.xlabel('Global Steps')
    plt.ylabel('Loss')
    plt.legend()
    plt.savefig('loss.png')

    best_model = BERT().to(device)

    load_checkpoint(modelPath, best_model)

    evaluate(best_model, testSet)
