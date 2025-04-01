import torch

def normalize_samples(samples):
    """
    samples are of shape [s, b, c, <dims>]
    """
    if samples.shape[2] == 1:
        return torch.sigmoid(samples)
    else:
        assert samples.shape[2] == 2
        return torch.nn.functional.softmax(samples, dim=2)
    
    
def normalize_batch(batch):
    """
    batches are of shape [b, c, <dims>]
    """
    if batch.shape[1] == 1:
        assert samples.shape[2] == 2
        return torch.sigmoid(batch)
    else:
        return torch.nn.functional.softmax(batch, dim=1)
    
    
def mle_batch(batch, dtype=torch.float32):
    """
    batches are of shape [b, c, <dims>]
    """
    if batch.shape[1] == 1:
        mle = (batch > 0.5).squeeze()
    else:
        mle = batch.argmax(dim=1)
        
    return mle.type(dtype)
        
        
def mle_samples(samples, dtype=torch.float32):
    """
    samples are of shape [s, b, c, <dims>]
    """
    if samples.shape[2] == 1:
        mle = (samples > 0.5).squeeze()
    else:
        mle = samples.argmax(dim=2)
        
    return mle.type(dtype)