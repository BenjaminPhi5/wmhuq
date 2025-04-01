import torch
from tqdm import tqdm
from trustworthai.utils.losses_and_metrics.evidential_bayes_risks import softplus_evidence, get_alpha, get_S, get_mean_p_hat
from trustworthai.journal_run.model_load.load_ssn import load_ssn
from trustworthai.journal_run.model_load.load_punet import load_p_unet
from trustworthai.journal_run.model_load.load_deterministic import load_deterministic
from trustworthai.journal_run.model_load.load_evidential import load_evidential
import os
from trustworthai.journal_run.evaluation.new_scripts.eval_helper_functions import load_best_checkpoint

MODEL_LOADERS = {
    "deterministic":load_deterministic,
    "mc_drop":load_deterministic,
    "evidential":load_evidential,
    "ssn":load_ssn,
    "punet":load_p_unet,
}

def punet_mean_and_samples(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    
    if y != None:
        y = y.cuda()
    
    model_raw(x.swapaxes(0,1).cuda(), y, training=False)
    mean = model_raw.sample(use_prior_mean=True).cpu()
    
    ind_samples = []
    for j in range(num_samples):
                ind_samples.append(model_raw.sample(testing=False).cpu())

    ind_samples = torch.stack(ind_samples, dim=0)
    
    return mean, ind_samples, None


def evid_mean(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    logits = model_raw(x.swapaxes(0,1).cuda()).cpu()
    evidence = softplus_evidence(logits)
    alpha = get_alpha(evidence)
    # print(alpha.shape)
    S = get_S(alpha)
    K = alpha.shape[1]
    mean_p_hat = get_mean_p_hat(alpha, S)
    
    return mean_p_hat, None, None

def deterministic_mean(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    x = x.swapaxes(0,1).cuda()
    mean = model_raw(x).cpu()
    return mean, None, None

def ssn_mean_and_samples(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    remove_last=False
    if num_samples %2 != 0:
        num_samples +=1
        remove_last=True
    mean, sample = model_raw.mean_and_sample(x.swapaxes(0,1).cuda(), num_samples=num_samples, temperature=1)
    
    if remove_last:
        sample = sample[:-1]
    
    return mean, sample, None

def mc_drop_mean_and_samples(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    x = x.swapaxes(0,1).cuda()
    # mean = model_raw(x)
    model_raw.train()
    ind_samples = []
    for j in range(num_samples):
        ind_samples.append(model_raw(x).cpu())
    model_raw.eval()
    ind_samples = torch.stack(ind_samples, dim=0)
    mean = ind_samples.mean(dim=0)
    return mean, ind_samples, None

def ensemble_mean_and_samples(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    args = inputs['args']
    samples = []
    x = x.swapaxes(0,1).cuda()
    
    for i in range(min(num_samples, 10)):
        model_dir = os.path.join(args.ckpt_dir, f"ens{i}_cv{args.cv_split}")  
        print("model dir: ", model_dir)
        model_raw, loss, val_loss = MODEL_LOADERS[args.model_type](args)
        model = load_best_checkpoint(model_raw, loss, model_dir, punet=False)
        samples.append(model.cuda()(x).cpu())
    
    samples = torch.stack(samples)
    mean = samples.mean(dim=0)
    
    if num_samples > 10:
        samples = [None for _ in range(len(samples))]
    
    return mean, samples, None

def ssn_ensemble_mean_and_samples(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    args = inputs['args']
    x = x.swapaxes(0,1).cuda()
    collected_samples = 0
    samples = []
    means = []
    collected_means = False
    while collected_samples < num_samples:
        for i in range(10):
            collected_samples += 1
            model_dir = os.path.join(args.ckpt_dir, f"ssn_ens{i}_cv{args.cv_split}")  
            print("model dir: ", model_dir)
            model_raw, loss, val_loss = MODEL_LOADERS[args.model_type](args)
            model = load_best_checkpoint(model_raw, loss, model_dir, punet=False)
            mean, sample = model_raw.mean_and_sample(x, num_samples=2, temperature=1)
            samples.append(sample[0])
            if not collected_means:
                means.append(mean)
            if collected_samples == num_samples:
                break
                
        collected_means = True
        
    samples = torch.stack(samples)
    print("total samples: ", samples.shape[0])
    mean = torch.stack(means).mean(dim=0)
    
    return mean, samples, None

def get_means_and_samples(model_raw, eval_ds, num_samples, model_func, args={}):
    means = []
    samples = []
    miscs = []

    model_raw.eval()
    for i, data in enumerate(tqdm(eval_ds, position=0, leave=True)):
        x = data[0]
        y = data[1]
        inputs = {"model":model_raw, "x":x, "y":y, "num_samples":num_samples, "args":args}
        with torch.no_grad():
            mean, sample, misc = model_func(inputs)
            means.append(mean.cpu())
            if sample != None:
                sample = sample.cpu()
            samples.append(sample)
            miscs.append(misc)
            
    return means, samples, miscs

def reorder_samples(sample):
    sample = sample.cuda()
    slice_volumes = sample.argmax(dim=2).sum(dim=(-1, -2))
    slice_volume_orders = torch.sort(slice_volumes.T, dim=1)[1]
    
    # rearrange the samples into one...
    new_sample = torch.zeros(sample.shape).to(sample.device)
    for i, slice_volumes_orders in enumerate(slice_volume_orders):
        for j, sample_index in enumerate(slice_volumes_orders):
            new_sample[j][i] = sample[sample_index][i]
            
    return new_sample.cpu()

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


def entropy_map_from_mean(mean, do_normalize=True, **kwargs):
    "samples is of shape samples, batch size, channels, image dims  [s, b, c *<dims>]"
    if mean.shape[1] == 1:
        raise ValueError("not implemented for implicit background class")
    else:
        assert mean.shape[1] == 2
    
    if do_normalize:
        probs = torch.nn.functional.softmax(mean, dim=1)
    else:
        probs = mean
    ent_map = torch.sum(-probs * torch.log(probs+1e-30), dim=1)

    return ent_map

UNCERTAINTY_MAP_GENERATORS = {
    "deterministic":entropy_map_from_mean,
    "mc_drop":entropy_map_from_samples,
    "evidential":entropy_map_from_mean,
    "evidential_aleatoric":None,
    "ssn":entropy_map_from_samples,
    "punet":entropy_map_from_samples,
    "ssn_ens":entropy_map_from_samples,
    "ens":entropy_map_from_samples,
    "softmax_ent":entropy_map_from_mean,
    "ind":entropy_map_from_samples,
}

def get_uncertainty_maps(means, samples, misc, args):
    ent_maps = []
    print(args.uncertainty_type)
    umap_func = UNCERTAINTY_MAP_GENERATORS[args.uncertainty_type]
    print("generating uncertainty maps")
    for idx in tqdm(range(len(means)), position=0, leave=True):
        umap_params = {"mean":means[idx], "samples":samples[idx], "misc":misc[idx], "do_normalize":args.uncertainty_type != "evidential"}
        ent_maps.append(umap_func(**umap_params))
    return ent_maps