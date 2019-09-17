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