# -*- coding: utf-8 -*-
"""
This code is for a multitask ner task

"""


import os
import torch
from torch import nn
import math
import numpy as np
import pandas as pd
import matplotlib as plot
import nltk
import gensim
from gensim.models import Word2Vec
import random
from sklearn.metrics import accuracy_score
from torchcrf import CRF
from allennlp.commands.elmo import ElmoEmbedder
from sklearn import metrics
import pickle
import time
from IPython.display import clear_output

torch.cuda.is_available()
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
if device.type == 'cuda':
    print(torch.cuda.get_device_name(0))
    print('Memory Usage:')
    print('Allocated:', round(torch.cuda.memory_allocated(0)/1024**3,1), 'GB')
    print('Cached:   ', round(torch.cuda.memory_cached(0)/1024**3,1), 'GB')

basepath = './cner_multitask/'

def update_progress(progress):
    bar_length = 20
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
    if progress < 0:
        progress = 0
    if progress >= 1:
        progress = 1
    block = int(round(bar_length * progress))
    clear_output(wait = True)
    text = "Progress: [{0}] {1:.1f}%".format( "#" * block + "-" * (bar_length - block), progress * 100)
    print(text)

path = 'data/i2b2_seqs/'
filename = 'train_seqs_cats'
train_seqs, train_cats = pickle.load(open(basepath+path+filename+ '.pkl', 'rb'))
path = 'data/i2b2_seqs/'
filename = 'test_seqs_cats'
test_seqs, test_cats = pickle.load(open(basepath+path+filename+ '.pkl', 'rb'))

path = 'data/i2b2_seqs/'
filename = 'train_emb'
train_emb = pickle.load(open(basepath+path+filename+ '.pkl', 'rb'))
path = 'data/i2b2_seqs/'
filename = 'test_emb'
test_emb = pickle.load(open(basepath+path+filename+ '.pkl', 'rb'))

tmp = []
for seq_cat in train_cats:
    tmp +=seq_cat
labels = list(set(tmp))

def label2idx(l,labels):
    return labels.index(l)
def idx2label(idx,labels):
    return labels[idx]
def cat2tensor(seq_cat,labels):
    # seq's cat convert to idx tensor ([I,I,B,O]=>[0,0,2,1])
    tensor = torch.zeros((1,len(seq_cat)),dtype=torch.long)
    for i,cat in enumerate(seq_cat):
        cat_id = label2idx(cat,labels)
        tensor[0,i] = cat_id
    return tensor
def prop2cat(tensor,labels):
    prop,max_cat_index = tensor.topk(1)
    max_cat_index = max_cat_index.item()
    return idx2label(max_cat_index,labels),max_cat_index

train_labels = []
for seq_idx,seqcats in enumerate(train_cats):
    train_labels.append(cat2tensor(seqcats,labels))

a_seqs = train_seqs+test_seqs
max_len = np.max([len(se) for se in a_seqs])

def generate_batch(seqs_set,seq_cats_label,batch_size):
    batch_units_i = np.random.randint(low=0,high=len(seqs_set),size=batch_size).tolist()
    seqs = [seqs_set[i] for i in batch_units_i]
    batch_cats = [torch.LongTensor(seq_cats_label[u]) for u in batch_units_i]
    
    len_units =  torch.LongTensor([len(u) for u in seqs])
    
    seq_tensor = torch.zeros((batch_size, len_units.max(),1024),dtype=torch.float).to(device)
    mask  = torch.zeros((batch_size, len_units.max()),dtype=torch.long).to(device)
    for idx, (seq_idx,seq, seqlen) in enumerate(zip(batch_units_i,seqs, len_units)):
        #seq_vec = train_emb[seq_idx][0][2]
        #seq_vec = torch.cat((train_emb[seq_idx][0][0],train_emb[seq_idx][0][2]),dim=1)
        seq_vec = train_emb[seq_idx][0].mean(dim=0)
        seq_tensor[idx, :seqlen] = seq_vec
        mask[idx,:seqlen] = 1
    sorted_len_units, perm_idx = len_units.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]
    mask = mask[perm_idx]   
    sorted_batch_cats = []
    for idx in perm_idx:
        sorted_batch_cats.append(batch_cats[idx])
    packed_input = nn.utils.rnn.pack_padded_sequence(seq_tensor, sorted_len_units, batch_first=True)
    return seq_tensor,sorted_batch_cats,sorted_len_units,packed_input,mask
batch_size = 3
seq_tensor,batch_cats,sorted_len_units,packed_input,mask = generate_batch(train_seqs,train_labels,batch_size)

class BiLSTMEncoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size=1, num_layers = 1, bi=True):
        super(BiLSTMEncoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.lstm = nn.LSTM(input_size=self.input_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.num_layers,
                            dropout=0.5,
                            batch_first=True,bidirectional=bi)
        self.linear = nn.Linear(self.hidden_size*2,self.output_size)
    def forward(self,inp,hn):
        lstm_out,(hn_,st_) = self.lstm(inp,hn)
        lstm_out = nn.utils.rnn.pad_packed_sequence(lstm_out,batch_first=True)
        return lstm_out[0],hn_,st_
    def initH0C0(self):
        return (torch.zeros((2*self.num_layers,self.batch_size,self.hidden_size),dtype=torch.float32,device=device),
                torch.zeros((2*self.num_layers,self.batch_size,self.hidden_size),dtype=torch.float32,device=device))

class BiLSTMDecoder(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, max_len,batch_size=1, num_layers = 1, bi=True):
        super(BiLSTMDecoder,self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.num_layers = num_layers
        self.batch_size = batch_size
        self.max_len = max_len
        self.mutihead_attention = nn.MultiheadAttention(self.input_size,num_heads = 2)
        self.linear = nn.Linear(self.hidden_size*2,self.output_size)
        self.seg_linear = nn.Linear(self.hidden_size*2,2)
        self.crf = CRF(self.output_size,batch_first=True)
        self.softmax = nn.LogSoftmax(dim=2)
    def generate_masked_labels(self,observed_labels,mask):
        masked_labels = torch.zeros((mask.size(0),mask.size(1)),dtype=torch.long).to(device)
        for i in range(mask.size(0)):
            masked_labels[i,:len(observed_labels[i][0])] = observed_labels[i][0]
        return masked_labels
    def forward(self,encoder_outputs,hn,batch_cats,mask):
        x = encoder_outputs.permute(1,0,2)
        attn_output, attn_output_weights = self.mutihead_attention(x,x,x)
        z = attn_output.permute(1,0,2)
        decoder_ipts  = nn.functional.relu(z)
        
        fc_out = self.linear(decoder_ipts)
        seg_weights = self.seg_linear(decoder_ipts)
        #fc_out = self.linear(encoder_outputs)
        masked_labels = self.generate_masked_labels(batch_cats,mask)
        mask = mask.type(torch.uint8).to(device)
        crf_loss = self.crf(fc_out,masked_labels,mask,reduction='token_mean')
        out = self.crf.decode(fc_out)
        seg_out = self.softmax(seg_weights)
        return out,seg_out,-crf_loss

hidden_size = 64
output_size = len(labels)
decoder_insize = hidden_size*2
batch_size = 2
embedding_dim = 1024
num_layers=2
encoder = BiLSTMEncoder(embedding_dim,hidden_size,batch_size=batch_size,num_layers=num_layers).to(device=device)
seq_tensor,batch_cats,sorted_len_units,packed_input,mask = generate_batch(train_seqs,train_labels,batch_size)
h0c0 = encoder.initH0C0()
eout,hn,st = encoder(packed_input.to(device=device),h0c0)
decoder = BiLSTMDecoder(decoder_insize,hidden_size,output_size,max_len=max_len,batch_size=batch_size,num_layers=num_layers).to(device=device)
out,seg_out,crf_loss = decoder(eout,(hn,st),batch_cats,mask)


for i,l in enumerate(labels):
    print(i,l)

entity_labels = [i for i in range(13) if labels[i].startswith('B') or labels[i].startswith('E') or labels[i].startswith('S') or labels[i].startswith('I')]
seed = [35899,54377,66449,77417,29,229,1229,88003,99901,11003]
random_seed = seed[9]
random.seed(random_seed)
np.random.seed(random_seed)
torch.manual_seed(random_seed)
hidden_size = 256
output_size = len(labels)
decoder_insize = hidden_size*2
batch_size = 64
num_layers = 2
encoder = BiLSTMEncoder(embedding_dim,hidden_size,batch_size=batch_size,num_layers=num_layers).to(device=device)
decoder = BiLSTMDecoder(decoder_insize,hidden_size,output_size,batch_size=batch_size,max_len=max_len,num_layers=num_layers).to(device=device)
criterion = nn.NLLLoss()
lr = 1e-3
encoder_optimizer = torch.optim.Adam(encoder.parameters(), 
                             lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
decoder_optimizer = torch.optim.Adam(decoder.parameters(), 
                             lr=lr, betas=(0.9, 0.999), eps=1e-08, weight_decay=0, amsgrad=False)
#16315/batch_size,128,255,510
plot_every=255
current_loss = 0
all_losses=[]
target_num = 0
iters = 0
loss = 0
epochs = 100
n_iters = epochs*plot_every
acc = 0

start = time.clock()
for iiter in range(n_iters):
    #idx = random.randint(0,len(train_seqs)-1)
    seq_tensor,batch_cats,sorted_len_units,packed_input,mask = generate_batch(train_seqs,train_labels,batch_size)
    h0c0 = encoder.initH0C0()
    eout,hn,st = encoder(packed_input.to(device=device),h0c0)
    crf_out,seg_out,crf_loss = decoder(eout,(hn,st),batch_cats,mask)
    seg_pred = torch.zeros((sorted_len_units.sum(),2),dtype=torch.float,device=device)
    seg_true = torch.zeros((sorted_len_units.sum()),dtype=torch.long,device=device)
    label_num = 0
    for b in range(batch_size):
        sout_ = seg_out[b]
        len_ = sorted_len_units[b]
        out_with_seg = sout_[:len_,:]
        for j in range(len_.item()):
            if idx2label(batch_cats[b][0][j],labels)!='O':
                seg_true[label_num] = 1
            else:
                seg_true[label_num] = 0
            seg_pred[label_num] = out_with_seg[j,:]
            label_num += 1
    
    seg_loss = criterion(seg_pred,seg_true)
    loss = crf_loss+0.2*seg_loss
    encoder_optimizer.zero_grad()
    decoder_optimizer.zero_grad()
    loss.backward()
    encoder_optimizer.step()
    decoder_optimizer.step()
    current_loss +=loss.item()
    label_pred = torch.zeros((sorted_len_units.sum()),dtype=torch.long,device=device)
    label_true = torch.zeros((sorted_len_units.sum()),dtype=torch.long,device=device)
    label_num = 0
    for b in range(batch_size):
        out_ = crf_out[b]
        len_ = sorted_len_units[b]
        out_with_label = out_[:len_]
        for j in range(len_.item()):
            label_true[label_num] = batch_cats[b][0][j]
            label_pred[label_num] = out_with_label[j]
            label_num += 1
    acc += metrics.f1_score(label_pred.tolist(),label_true.tolist(),average='micro',labels=entity_labels)
    if (iiter+1) % plot_every == 0:
        all_losses.append(current_loss / plot_every)
        print("epoch: %d | F1: %.3f%% | avg_loss: %.5f" % 
              (iiter+1,acc/plot_every*100,current_loss / plot_every))
        current_loss = 0
        acc = 0
        elapsed = (time.clock() - start)
        print("Time used:",elapsed)
        start = time.clock()

encoder = encoder.eval()
decoder = decoder.eval()
def run_test(test_seqs,test_cats):
    y_ = []
    y = []
    y_pred = []
    for idx,xs in enumerate(test_seqs):
        x = torch.zeros((1,len(xs),embedding_dim)).to(device)
        mask = torch.zeros((1,len(xs)),dtype=torch.long).to(device)
        mask[0] = 1
        x[0] =test_emb[idx][0].mean(dim=0)
        x = nn.utils.rnn.pack_padded_sequence(x, torch.tensor([x.shape[1]]), batch_first=True)
        ys = test_cats[idx]
        y_label = [cat2tensor(ys,labels).to(device)]
        h0c0 = (torch.zeros((2*num_layers,1,hidden_size),dtype=torch.float32,device=device),
                torch.zeros((2*num_layers,1,hidden_size),dtype=torch.float32,device=device))
        eout,hn,st = encoder(x.to(device=device),h0c0)
        out,seg_out,loss = decoder(eout,(hn,st),y_label,mask)
        y_.extend(out[0])
        y.extend(y_label[0][0].tolist())
        if (idx+1) % 100==0:
            update_progress(idx / len(test_seqs))
    update_progress(1)
    return y_,y

from sklearn.metrics import classification_report
y_a,ya = run_test(test_seqs,test_cats)
from sklearn import metrics
print(classification_report(ya, y_a,labels=entity_labels))

for i,l in enumerate(labels):
    print(i,l)

entity_num_true = 0
entity_num_pred = 0
begin_labels = ['B-test','B-problem','B-treatment']
single_labels = ['S-test','S-problem','S-treatment']
label_pairs = {'B-test':['I-test','E-test'],
               'B-problem':['I-problem','E-problem'],
               'B-treatment':['I-treatment','E-treatment']}
entities = ['test','problem','treatment']

y = ya
y_ = y_a

true_num = {
    'problem':0,#problem
    'test':0,#test
    'treatment':0#treatment
}

y = ya
y_ = y_a
true_entity = {}
i = 0
lentag= len(y)
while i<lentag:
    tag = idx2label(y[i],labels)
    if tag in begin_labels:
        ent = tag[2:]
        k=i+1
        tag_in = idx2label(y[k],labels)
        while tag_in != 'E-'+ent:
            k += 1
            tag_in = idx2label(y[k],labels)
        true_entity[(i,k)] = ent
    if tag in single_labels:
        ent = tag[2:]
        true_entity[(i,i)] = ent
    i+=1
for k,ent in true_entity.items():
    true_num[ent]+=1


pred_num = {
    'problem':0,#problem
    'test':0,#test
    'treatment':0#treatment
}
pred_entity = {}
i = 0
y_ = y_a
lentag= len(y_)
while i<lentag:
    tag = idx2label(y_[i],labels)
    if tag in begin_labels:
        ent = tag[2:]
        k=i+1
        tag_in = idx2label(y_[k],labels)
        while tag_in != 'E-'+ent:
            k += 1
            tag_in = idx2label(y_[k],labels)
        pred_entity[(i,k)] = ent
    if tag in single_labels:
        ent = tag[2:]
        pred_entity[(i,i)] = ent
    i+=1
for k,ent in pred_entity.items():
    pred_num[ent]+=1


exact_hitted = {
    'problem':0,#problem
    'test':0,#test
    'treatment':0#treatment
}
for k,ent in true_entity.items():
    if k in pred_entity.keys() and pred_entity[k]==ent:
        exact_hitted[ent]+=1


entities = ['test','problem','treatment']
recall = [exact_hitted[ent]/true_num[ent] for ent in entities]
precision = [exact_hitted[ent]/pred_num[ent] for ent in entities]
fp = np.mean([(2*precision[i]*recall[i])/(precision[i]+recall[i]) for i in range(len(entities)) ])

round(fp,4),round(np.mean(precision),4),round(np.mean(recall),4)

from datetime import datetime,timezone,timedelta
dt = datetime.utcnow()
dt = dt.replace(tzinfo=timezone.utc)
tzutc_8 = timezone(timedelta(hours=-4))
local_dt = dt.astimezone(tzutc_8)
path = 'results_model/'
filename = 'I2B2_mimic_elmo_'+local_dt.strftime("%Y%m%d_%H%M")+'-'+str(random_seed)
print(filename)
pickle.dump([y_a, ya], open(basepath+path+'results-'+filename+ '.pkl', 'wb'))
pickle.dump(labels, open(basepath+path+'idx_2_labels-'+filename+ '.pkl', 'wb'))
torch.save(encoder, basepath+path+'model_encoder-'+filename)
torch.save(decoder, basepath+path+'model_decoder-'+filename)
