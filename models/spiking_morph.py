import torch
import torch.nn as nn
import spikingjelly.clock_driven as snn


class LearnableStructuringElement(nn.Module):
    def __init__(self, size):
        super(LearnableStructuringElement, self).__init__()
        # Initializing the weights of the structuring element with random values
        self.weights = nn.Parameter(torch.randn(size, size))
        
    def forward(self):
        # Normalizing the weights to ensure they are non-negative and sum to 1
        normalized_weights = torch.nn.functional.softmax(self.weights.view(-1), dim=0).view(self.weights.size())
        return normalized_weights

class ErosionLayer(snn.layer):
    def __init__(self, size):
        super(ErosionLayer, self).__init__()
        self.struct_elem = LearnableStructuringElement(size)
    
    def erosion(self, spike_train):
        eroded_train = []
        struct_elem = self.struct_elem()
        for i, event in enumerate(spike_train):
            neighborhood = spike_train[i:i+struct_elem.shape[0], i:i+struct_elem.shape[1]]
            if (neighborhood * struct_elem).sum() == struct_elem.sum():
                eroded_train.append(event)
        return torch.stack(eroded_train, dim=0)

    def forward(self, x):
        return self.erosion(x)

class DilationLayer(snn.layer):
    def __init__(self, size):
        super(DilationLayer, self).__init__()
        self.struct_elem = LearnableStructuringElement(size)
    
    def dilation(self, spike_train):
        dilated_train = []
        struct_elem = self.struct_elem()
        for i, event in enumerate(spike_train):
            neighborhood = spike_train[i:i+struct_elem.shape[0], i:i+struct_elem.shape[1]]
            if event == 1:
                dilated_train.append(neighborhood + struct_elem)
        return torch.cat(dilated_train, dim=0)

    def forward(self, x):
        return self.dilation(x)

