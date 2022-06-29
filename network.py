import torch.nn as nn

class SilvaNet(nn.Module):
    def __init__(self):
        super(SilvaNet, self).__init__()
        
        self.conv_branch = nn.Sequential(
            nn.BatchNorm1d(1),
            nn.Conv1d(1, 96, 49),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.BatchNorm1d(96),
            nn.Conv1d(96, 128, 25),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.BatchNorm1d(128),
            nn.Conv1d(128, 256, 9),
            nn.ReLU(),
            nn.MaxPool1d(2),
            
            nn.BatchNorm1d(256),
            nn.Conv1d(256, 512, 9),
            nn.ReLU(),
            nn.MaxPool1d(2)
        )
        
        # self.lstm = nn.LSTM(N_CHANNELS_CNN, N_CHANNELS_CNN, 1)
        
        self.linear = nn.Sequential(nn.Dropout(0.5),
                                    nn.Linear(1536, 4096),
                                    nn.ReLU(),
                                    nn.Linear(4096, 4096),
                                    nn.ReLU(),
                                    nn.Linear(4096, 2),
                                    nn.ReLU(),
                                    nn.Softmax(1))
    
    
    def forward(self, x):
        x = x.unsqueeze(1)
        x_conv = self.conv_branch(x)
        x_conv = x_conv.view(x_conv.shape[0], -1)
        x_out = self.linear(x_conv)
        
        return(x_out)
            
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                nn.init.constant_(m.bias, 0)
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if isinstance(m, nn.BatchNorm1d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
