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