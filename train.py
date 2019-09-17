def train():
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
def save():
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
