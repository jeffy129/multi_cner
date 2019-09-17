def eval(): 
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