import torch.nn as nn
from abc import ABC, abstractmethod
from torch import no_grad
from tqdm import tqdm

class UncertaintyQuantificationModel(nn.Module, ABC):
    @abstractmethod
    def mean(self, x, *args, **kwargs):
        pass
    
    @abstractmethod
    def mean_and_sample(self, x, num_samples, rsample, temperature, *args, **kwargs):
        pass

    def sample_over_3Ddataset(self, dataset, num_samples, rsample, temperature=1, dataset_stride=1, to_cpu=False, *args, **kwargs):
        with no_grad():
            samples3d = []
            means3d = []
            # the model is 2d, so we have to play about with the axes
            skip = 0
            for data in tqdm(dataset):
                if skip % dataset_stride == 0:
                    x = data[0].cuda().swapaxes(0,1)
                    mean, samples = self.mean_and_sample(x, num_samples, rsample=rsample, temperature=temperature)
                    if to_cpu:
                        mean = mean.cpu()
                        samples = samples.cpu()
                    means3d.append(mean)
                    samples3d.append(samples)
                skip += 1
        
        return means3d, samples3d
    
    def no_grad_mean_and_sample(self, *args, **kwargs):
        with no_grad():
            return self.mean_and_sample(*args, **kwargs)