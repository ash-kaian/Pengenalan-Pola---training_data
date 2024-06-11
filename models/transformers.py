import torch
import torch.nn as nn
import torch.nn.functional as F

class CNNFormer(nn.Module):
    def __init__(self, feature_dim, dff=1024, num_head=1, num_layer=1, n_class=3, dropout=0.1, device='cpu'):
        super(CNNFormer, self).__init__()
        self.layer = num_layer
        self.conv = nn.Sequential(
            nn.Conv2d(feature_dim, 20, 2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 20, 5),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(20, 20, 3),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            nn.Dropout(p=dropout),
        )

        # Hidden dimension based on number of filters
        self.hidden_dim = 20
        self.MHA = nn.MultiheadAttention(embed_dim=self.hidden_dim, num_heads=num_head, bias=False, dropout=dropout).to(device)
        self.feed_forward = nn.Sequential(
            nn.Linear(self.hidden_dim, dff),
            nn.ReLU(),
            nn.Linear(dff, self.hidden_dim)
        )
        self.norm = nn.LayerNorm(self.hidden_dim)
        
        # Calculate flattened dimension
        self.flatten_dim = self.calculate_flatten_dim((3, 200, 200))
        self.lin_out = nn.Linear(self.flatten_dim, n_class)

    def calculate_flatten_dim(self, input_shape):
        dummy_input = torch.zeros(1, *input_shape)
        dummy_output = self.conv(dummy_input)
        return dummy_output.numel()

    def forward(self, x):
        x = self.conv(x)
        batch_size = x.size(0)
        x = x.view(batch_size, -1)
        x = self.lin_out(x)
        return x

# Create the model
def create_cnn_model(input_shape, num_classes):
    model = CNNFormer(feature_dim=input_shape[0], n_class=num_classes)
    return model
