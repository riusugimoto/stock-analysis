import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

import utils


class TorchNeuralNetClassifier(nn.Module):
    def __init__(
        self, 
        hidden_layer_sizes =[512, 256, 128,64],   ##############
        X=None,
        y=None,
        max_iter=10000,
        learning_rate=0.001,###############
        init_scale=0.1,###
        batch_size=64,###
        weight_decay=1e-4,
        device='cuda' if torch.cuda.is_available() else 'cpu',
    ):
        # This isn't really the typical way you'd lay out a pytorch module;
        # usually, you separate building the model and training it more.
        # This layout is like what we did before, though, and it'll do.
        super().__init__()

        self.hidden_layer_sizes = hidden_layer_sizes
        self.learning_rate = learning_rate
        self.max_iter = max_iter
        self.init_scale = init_scale
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device

        if X is not None and y is not None:
            self.fit(X, y)

    def cast(self, ary):
        # pytorch defaults everything to float32, unlike numpy which defaults to float64.
        # it's easier to keep everything the same,
        # and most ML uses don't really need the added precision...
        # you could use torch.set_default_dtype,
        # or pass dtype parameters everywhere you create a tensor, if you do want float64
        ary = ary / 255.0
        return torch.as_tensor(ary, dtype=torch.get_default_dtype(), device=self.device)

    # def convert_label(self, y):
    #     # input: a *numpy array* of integers between 0 and 9
    #     # output: a torch tensor in whatever format you'd like it to be

    #     # for right now, just convert to a float tensor and make shape [t, 1]
    #     return self.cast(y).unsqueeze(1)
    
    def convert_label(self, y):
    # Just cast to tensor of type long without unsqueezing
        return torch.as_tensor(y, dtype=torch.long, device=self.device)
    


    def output_shape(self, y):
        return 1

    def predicted_label(self, y):
        # input: a tensor of whatever your network outputs
        # output: a tensor of integers (hard prediction)

        # round our scalar output to the nearest integer
        return torch.round(torch.clamp(y, 0, 9)).squeeze(1).int()
        

    def loss_function(self):
        #return nn.MSELoss()
        return nn.CrossEntropyLoss()

    def init(self, weight):
        nn.init.normal_(weight, mean=0, std=self.init_scale)

    def nonlinearity(self):
        # return nn.Tanh()
        return nn.ReLU()




    def build(self, in_dim, out_dim):
        out_dim = 10
        layer_sizes = [in_dim] + list(self.hidden_layer_sizes) + [out_dim]

        layers = []
        for in_size, out_size in zip(layer_sizes[:-1], layer_sizes[1:]):
            lin = nn.Linear(in_size, out_size, device=self.device)
            self.init(lin.weight)
            layers.append(lin)
            layers.append(self.nonlinearity())

        layers.pop(-1)  # drop the final activation

        self.layers = nn.Sequential(*layers)



    def make_optimizer(self):
        return torch.optim.Adam(
            self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay
        )

    def forward(self, x):
        return self.layers.forward(self.cast(x))

    def fit(self, X, y):
        X = self.cast(X)
        y = self.convert_label(y)

        self.build(X.shape[1], self.output_shape(y))

        loss_fn = self.loss_function()
        self.optimizer = self.make_optimizer()

        for i in range(self.max_iter):
            # Not doing anything fancy here like early stopping, etc.
            self.optimizer.zero_grad()

            inds = torch.as_tensor(
                np.random.choice(X.shape[0], size=self.batch_size, replace=False)
            )
            yhat = self(X[inds])
            loss = loss_fn(yhat, y[inds])

            loss.backward()
            self.optimizer.step()

            if i % 500 == 0:
                print(f"Iteration {i:>10,}: loss = {loss:>6.3f}")

                

    # def predict(self, X):
    #     # hard class predictions (for soft ones, just call the model like the Z line below)
    #     with torch.no_grad():
    #         Z = self(X)
    #         return self.predicted_label(Z).cpu().numpy()
        
    def predict(self, X):
        with torch.no_grad():
            outputs = self(X)
            return torch.argmax(outputs, dim=1).cpu().numpy()



# class Convnet(TorchNeuralNetClassifier):
#     def __init__(self, **kwargs):
#         # default the hidden_layer_sizes thing to None
#         kwargs.setdefault("hidden_layer_sizes", None)
#         super().__init__(**kwargs)

#     def forward(self, x):
#         # hardcoding the shape of the data here, kind of obnoxiously.
#         # normally we wouldn't have flattened the data in the first place,
#         # just passed it in with shape e.g. (batch_size, channels, height, width)
#         assert len(x.shape) == 2 and x.shape[1] == 784
#         unflattened = x.reshape((x.shape[0], 1, 28, 28))
#         return super().forward(unflattened)

#     def build(self, in_dim, out_dim):
#         # assign self.layers to an nn.Sequential with some Conv2d (and other) layers in it
#         raise NotImplementedError()




class Convnet(TorchNeuralNetClassifier):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
      


    def forward(self, x):
        # CNN input shape [N, C, H, W]
        if len(x.shape) == 2: 
            x = x.reshape(-1, 1, 28, 28) 
        return super().forward(x)

    def build(self, in_dim=None, out_dim=10):  
    
        self.layers = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1),  # Output: [N, 32, 28, 28]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [N, 32, 14, 14]
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),  # Output: [N, 64, 14, 14]
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Output: [N, 64, 7, 7]
            nn.Flatten(),  
            nn.Linear(64 * 7 * 7, 128), 
            nn.ReLU(),
            nn.Dropout(0.5),  
            nn.Linear(128, 10)  # 10 outputs for digits
        )
