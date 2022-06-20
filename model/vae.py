import torch; torch.manual_seed(0)
import torch.nn as nn
import torch.nn.functional as F
import torch.utils
import torch.distributions

class VariationalEncoder(nn.Module):
    def __init__(self):
        super(VariationalEncoder, self).__init__()
        self.linear1 = nn.Linear(115, round(115*0.75))
        self.linear2 = nn.Linear(round(115*0.75), round(115*0.50))
        self.linear3 = nn.Linear(round(115*0.50), round(115*0.33))
        self.linear4 = nn.Linear(round(115*0.33), round(115*0.25))
        self.linear5 = nn.Linear(round(115*0.33), round(115*0.25))

        self.N = torch.distributions.Normal(0, 1)
        self.kl = 0

    def forward(self, x):
        x = F.relu(self.linear1(x))
        x = F.relu(self.linear2(x))
        x = F.relu(self.linear3(x))
        mu =  self.linear4(x)
        sigma = torch.exp(self.linear5(x))
        z = mu + sigma*self.N.sample(mu.shape)
        self.kl = (sigma**2 + mu**2 - torch.log(sigma) - 1/2).sum()
        return z


class VariationalDecoder(nn.Module):
    def __init__(self):
        super(VariationalDecoder, self).__init__()
        self.linear1 = nn.Linear(round(115*0.25), round(115*0.33))
        self.linear2 = nn.Linear(round(115*0.33), round(115*0.50))
        self.linear3 = nn.Linear(round(115*0.50), round(115*0.75))
        self.linear4 = nn.Linear(round(115*0.75), 115)

    def forward(self, z):
        z = F.relu(self.linear1(z))
        z = F.relu(self.linear2(z))
        z = F.relu(self.linear3(z))
        return self.linear4(z)


class VariationalAutoEncoder(nn.Module):
    def __init__(self):
        super(VariationalAutoEncoder, self).__init__()
        self.encoder = VariationalEncoder()
        self.decoder = VariationalDecoder()
   
    def forward(self, x):
        encode = self.encoder(x)
        decode = self.decoder(encode)
        return decode
