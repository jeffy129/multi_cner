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
    
def pre_process():
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
    return ([],[])
