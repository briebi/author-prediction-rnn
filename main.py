import re
import json
import sys
import collections
import os
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# single-direction RNN, optionally tied embeddings (given in class code)
class Emb_RNNLM(nn.Module):
    def __init__(self, params, use_LSTM=True):
        super(Emb_RNNLM, self).__init__()
        self.vocab_size = params['vocab_size']
        self.d_emb = params['d_emb']
        self.n_layers = params['num_layers']
        self.d_hid = params['d_hid']
        self.embeddings = nn.Embedding(self.vocab_size, self.d_emb)
        self.use_LSTM = use_LSTM
        if use_LSTM:
            print('Using LSTM model')
            self.i2R = nn.LSTM(self.d_emb, self.d_hid, batch_first=True, num_layers=self.n_layers) #input to recurrent layer, default nonlinearity is tanh
        else:
            # input to recurrent layer, default nonlinearity is tanh
            self.i2R = nn.RNN(
                self.d_emb, self.d_hid, batch_first=True, num_layers = self.n_layers
            )
        # recurrent to output layer
        self.R2o = nn.Linear(self.d_hid, self.vocab_size)

    def forward(self, train_datum):
        embs = torch.unsqueeze(self.embeddings(train_datum), 0)
        if self.use_LSTM:
            output, (hidden, context) = self.i2R(embs)
        else:
            output, hidden = self.i2R(embs)
        return self.R2o(output)

# getting list of words, enurmerating them, and partitioning sentences
verbose = False
sentences = collections.defaultdict(lambda: [])
models = {}
book_titles = ['wuthering_sentences_shuf_6379', 'room_with_a_sentences_shuf_6379',
               'great_expectations_sentences_shuf_6379', 'emma_sentences_shuf_6379']
train_sent_as_ind = {}
test_sent_as_ind = {}

# list containing all words from all books
master_words =[]
for book_title in book_titles:
    if os.path.isfile(book_title):
        print('Processing file', book_title)
        with open(book_title, 'r') as f0:
            sentence_buffer = []
            for i, line in enumerate(f0.readlines()):
                if i % 1000 == 0:
                    print('Processed', i, 'lines.')

                line = line.rstrip()
                if len(line) < 1: continue
                if line[0] == '[': continue

                line = re.sub(r'([\.,;:!\?”])', r' \1', line)
                line = re.sub(r'(“)', r'\1 ', line)
                line = re.sub(r'[_‘]', '', line)
                line = re.sub('—', ' ', line)
                lal = line.split()
                for wd in lal:
                    sentence_buffer.append(wd.lower())
                    if wd not in master_words:
                        # adding words to master list
                        master_words.append(wd.lower())
                    if wd in ['.', '!', '?', ':', ';']:
                        sentences[book_title].append(sentence_buffer)
                        sentence_buffer = []
    else:
        print('No file found with name', book_title)
        exit()

# printing some words from master list
total_words = len(master_words)
print('Total Number of Words:', total_words)
print('Sample Words from Master List:')
for w in master_words[:25]:
    print(w)

# enumerating all the words
wd2ix = {}
ix2wd = {}

for i, word in enumerate(master_words):
    wd2ix[word] = i
    ix2wd[str(i)] = word
    if verbose and i < 100: print(word)

for book_title in book_titles:
    random.shuffle(sentences[book_title])
    num_sentences = len(sentences[book_title])

    # partitioning sets: training = 80%, test = 20%
    partition_point = int(num_sentences * 0.2)
    test_sentences = sentences[book_title][:partition_point]
    train_sentences = sentences[book_title][partition_point:]

    train_sent_as_ind[book_title] = [torch.LongTensor([wd2ix[w] for w in sent]) for sent in train_sentences]
    test_sent_as_ind[book_title] = [torch.LongTensor([wd2ix[w] for w in sent]) for sent in test_sentences]

    params = {'vocab_size': total_words, 'd_emb': 128, 'num_layers': 1, 'd_hid': 128, 'lr': 0.0003, 'epochs': 5}

    # creating a model for each book
    models[book_title] = Emb_RNNLM(params)

    # printing some sentences from each book to see if they are done correctly
    print("Sample Sentences from ", book_title, ":")
    for s in sentences[book_title][:10]:
        print(s)

# TRAINING
for book_title in book_titles:
    print("Training model for", book_title)
    criterion = nn.CrossEntropyLoss(ignore_index=0)
    optimiser = torch.optim.Adam(models[book_title].parameters(), lr=params['lr'])
    models[book_title].train()

    for epoch in range(10):
        ep_loss = 0

        # shuffling training set
        random.shuffle(train_sent_as_ind[book_title])

        # using random sample of training set
        fraction = 0.25
        subset_size = int(len(train_sent_as_ind[book_title]) * fraction)
        subset_data = random.sample(train_sent_as_ind[book_title], subset_size)
        random.shuffle(subset_data)

        for j, train_datum in enumerate(subset_data):
            if len(train_datum) < 4:
                continue
            preds = models[book_title](train_datum)
            preds = preds[:, :-1, :].contiguous().view(-1, params['vocab_size'])
            targets = torch.unsqueeze(train_datum, 0)
            targets = targets[:, 1:].contiguous().view(-1)

            loss = criterion(preds, targets)
            if torch.isnan(loss):
                print(train_datum)
            loss.backward()
            optimiser.step()
            optimiser.zero_grad()
            ep_loss += loss.detach()
            if j > 0 and j % 1000 == 0:
                print('processed', j, 'training examples')
        # printing results
        print('epoch', epoch, 'epoch loss', ep_loss / len(subset_data))

# TESTING: each model on test sentences from each book
# set models to evaluation mode
for book_title in book_titles:
    models[book_title].eval()

with torch.no_grad():
    # book_title = testing sentences we are looking at
    for book_title in book_titles:
        print('Testing sentences for ', book_title)

        # holds the total for probabilities for all the sentences in the given training set, for each model
        log_probs = {title: 0 for title in book_titles}
        num_sent = len(test_sent_as_ind[book_title])
        for sent in test_sent_as_ind[book_title]:
            # holds sum of the probabilites for one sentence across all four models
            sum_of_sent_log_probs = 0
            sentence_log_probs = {title: 0 for title in book_titles}

            # author = model we are looking at
            # calculating the probablity for a sentence given the model
            for author in book_titles:
                preds = models[author](sent)
                preds = preds[:, :-1, :].contiguous().view(-1, params['vocab_size'])

                # per-word log probability and aggregate for the sentence
                log_prob = 0
                for idx, word in enumerate(sent[1:]):
                    pred_probs = F.log_softmax(preds[idx], dim=0)  # convert to log probs
                    log_prob += pred_probs[word].item()  #  use the log prob of the target word

                # aggregating the probability for the sentence given the model
                sum_of_sent_log_probs += log_prob / len(sent)
                sentence_log_probs[author] = log_prob / len(sent)

            # p(s): averaging log probabilities for each model (author)
            avg_log_prob = sum_of_sent_log_probs / len(book_titles)

            # p(a|s): using Bayes' Rule and accumulate for each author
            for author in book_titles:
                # default if avg_log_prob is zero
                if avg_log_prob == 0:
                    prob = 0.25
                else:
                    # Bayes' Rule application: p(a|s) = p(s|a) * p(a) / p(s)
                    prob = 0.25 * exp(sentence_log_probs[author] - avg_log_prob)
                # summing up the probabilties of each sentence for the model
                log_probs[author] += prob
        # getting the average probability per sentence for each model
        for author in book_titles:
            log_probs[author] /= num_sent

        # print results
        print(f"\nPredicted average probabilities for book: '{book_title}'")
        for author, prob in log_probs.items():
            print(f"  Model {author}: {prob:.4f}")
