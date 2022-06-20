import torch.nn as nn

class MultiLayerPerceptron(nn.Module):
    def __init__(self):
        super(MultiLayerPerceptron, self).__init__()
        self.layers = nn.Sequential(
            nn.Linear(115, round(115*0.75)+1),
            #nn.ReLU(),
            nn.Sigmoid(),
            nn.Linear(round(115*0.75)+1, 1)
        )
        
    def forward(self, x):
        layers = self.layers(x)
        return layers
