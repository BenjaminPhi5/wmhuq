import torch

def entropy_map_from_samples(samples, do_normalize=True):
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


def entropy_map_from_samples_implicit(samples, do_normalize):
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