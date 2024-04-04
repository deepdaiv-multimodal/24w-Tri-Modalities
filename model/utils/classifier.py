import torch 
import torch.nn as nn
import torch.nn.functional as F 

class Classifier(nn.Module):
    def __init__(self, latent_dim):
        super(Classifier, self).__init__()

        self.layer_1 = nn.Linear(latent_dim, 256)
        self.layer_2 = nn.Linear(256, 128)
        self.layer_3 = nn.Linear(128, 20)
    
    def forward(self, x):
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        x = F.relu(x)
        x = self.layer_3(x)
<<<<<<< HEAD
        return x

=======
        return x
>>>>>>> 764b33b3e8c429784f721b3f7139f6b5f58782c0
