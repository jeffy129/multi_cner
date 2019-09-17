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