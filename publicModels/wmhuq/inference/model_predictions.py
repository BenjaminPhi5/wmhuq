import torch
import os

def ssn_ensemble_mean_and_samples(config_models, x, num_samples, device):
    x = x.swapaxes(0,1)
    collected_samples = 0
    samples = []
    means = []
    collected_means = False
    while collected_samples < num_samples:
        for i in range(10):
            collected_samples += 1
            model_i = config_models[i]#.to(device)
            
            with torch.no_grad():
                mean, sample = model_i.mean_and_sample(x, num_samples=2, temperature=1)
                # model_i = model_i.cpu()
            samples.append(sample[0])
            if not collected_means:
                means.append(mean)
            if collected_samples == num_samples:
                break
                
        collected_means = True
        
    samples = torch.stack(samples)
    mean = torch.stack(means).mean(dim=0)
    
    return mean, samples

def entropy_map_from_samples(samples, do_normalize=True, **kwargs):
    "samples is of shape samples, batch size, channels, image dims  [s, b, c *<dims>]"
    if samples.shape[2] == 1:
        return entropy_map_from_samples_implicit(samples, do_normalize)
    else:
        assert samples.shape[2] == 2
    
    if do_normalize:
        probs = torch.nn.functional.softmax(samples, dim=2)
    else:
        probs = samples

    pic = torch.mean(probs, dim=0)
    ent_map = torch.sum(-pic * torch.log(pic+1e-30), dim=1)

    return ent_map


def entropy_map_from_samples_implicit(samples, do_normalize, **kwargs):
    if do_normalize:
        probs = torch.sigmoid(samples)
    else:
        probs = samples
        
    pic = torch.mean(probs, dim=0)
    ent_map = (
        (-pic * torch.log(pic + 1e-30)) 
        + (-(1-pic) * torch.log((1-pic) + 1e-30))
    )
    return ent_map.squeeze()

def reorder_samples(sample):
    slice_volumes = sample.argmax(dim=2).sum(dim=(-1, -2))
    slice_volume_orders = torch.sort(slice_volumes.T, dim=1)[1]
    
    # rearrange the samples into one...
    new_sample = torch.zeros(sample.shape).to(sample.device)
    for i, slice_volumes_orders in enumerate(slice_volume_orders):
        for j, sample_index in enumerate(slice_volumes_orders):
            new_sample[j][i] = sample[sample_index][i]
            
    return new_sample
