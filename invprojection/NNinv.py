"""
Code according to Espadoto, M., Rodrigues, F. C. M., Hirata, N. S. T., & Hirata Jr, R. (2019). Deep Learning Inverse Multidimensional Projections. Proc. EuroVA, 5.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np



#### Pytorch version NNinv
class NNinv_net(nn.Module):
    def __init__(self, out_dim, dims=[2048, 2048, 2048, 2048], bottleneck_dim=2):
        super(NNinv_net, self).__init__()

        self.network = nn.Sequential(
            nn.Linear(bottleneck_dim, dims[0]),
            nn.ReLU(),
            nn.Linear(dims[0], dims[1]),
            nn.ReLU(),
            nn.Linear(dims[1], dims[2]),
            nn.ReLU(),
            nn.Linear(dims[2], dims[3]),
            nn.ReLU(),
            nn.Linear(dims[3], out_dim),
            nn.Sigmoid(),
    )

    def forward(self, x):
        return self.network(x)

class NNinv_torch:
    ### wrapper for sklearn API
    def __init__(self, dims=(128, 256, 512, 1024), bottleneck_dim=2):
        self.model = None
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print('device: ', self.device)
        self.dims = dims
        self.bottleneck_dim = bottleneck_dim
        # torch.manual_seed(42)
        self.bias = 0.01

    def create_model(self, n_dim):
        model = NNinv_net(n_dim, self.dims, self.bottleneck_dim)
        model.to(self.device)
        self.optimizer = optim.Adam(model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        return model

    def fit(self, X_2d, X, epochs=150, batch_size=128, verbose=True, early_stop=False, patience=5, **kwargs):
        ## no continuous training for the purpose of evaluation
        self.reset()
        ######################

        self.scaler = MinMaxScaler()
        X_2d = self.scaler.fit_transform(X_2d)
        X_2d = torch.tensor(X_2d, dtype=torch.float32).to(self.device)
        X = torch.tensor(X, dtype=torch.float32).to(self.device)

        if self.model is None:
            self.model = self.create_model(X.shape[1])

        #### original init ####################################################
        # for m in self.model.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        #         nn.init.constant_(m.bias, self.bias)
        ############### 

        for epoch in range(epochs):
            loss_per_epoch = 0
            for i in range(0, len(X_2d), batch_size):
                batch_X_2d = X_2d[i:i+batch_size]
                batch_X = X[i:i+batch_size]
                self.optimizer.zero_grad()
                output = self.model(batch_X_2d)
                loss = self.loss_fn(output, batch_X)
                loss.backward()
                self.optimizer.step()
                loss_per_epoch += loss.item() * batch_X.shape[0]
            if verbose:
                loss_per_epoch /= len(X_2d)
                print(f'Epoch {epoch+1}/{epochs}, Loss: {loss_per_epoch:.8f}') 

    def transform(self, X_2d, batch_size=128, **kwargs):
        with torch.no_grad():
            X_2d = self.scaler.transform(X_2d)
            X_recon = []
            for i in range(0, len(X_2d), batch_size):
                batch_X_2d = torch.tensor(X_2d[i:i+batch_size], dtype=torch.float32).to(self.device)
                X_recon.append(self.model(batch_X_2d).cpu().numpy())
            return np.concatenate(X_recon)
    
    def inverse_transform(self, X_2d, batch_size=128, **kwargs):
        return self.transform(X_2d, batch_size)

    def reset(self):
        self.model = None

    def find_z(self, dummy):
        return dummy


