from tqdm import tqdm
import torch
from twaibrain.braintorch.utils.resize import crop_or_pad_dims
import os
from twaibrain.brainexperiments.run_nnUNet_v2.evaluation.eval_helper_functions import load_best_checkpoint

def deterministic_mean(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    x = x.cuda()
    # print(x.shape)
    mean = model_raw(x)[0].squeeze(0).cpu()
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
    mean, sample = model_raw.mean_and_sample(x.cuda(), num_samples=num_samples)
    # print(mean.shape)
    # print(sample.shape)
    
    if remove_last:
        sample = sample[:-1]

    return mean, sample, None

<<<<<<< HEAD
def ssn_ensemble_mean_and_samples(inputs):
    model_set=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    args = inputs['args']
    collected_samples = 0
    samples = []
    means = []
    collected_means = False
    while collected_samples < num_samples:
        for i in range(10):
            collected_samples += 1
            # model_dir = os.path.join(args.ckpt_dir, f"nnunet2D_ssnV0_ens{i}_cv{args.cv_split}")  
            # print("model dir: ", model_dir)
            # ckpt = sorted([f for f in os.listdir(model_dir) if f.endswith(".ckpt")])[-1]
            # model = load_best_checkpoint(model_raw, None, None, None, os.path.join(model_dir, ckpt))
            model = model_set[i]
            
            mean, sample = model.mean_and_sample(x.cuda(), num_samples=2)
            samples.append(sample[0])
            if not collected_means:
                means.append(mean)
            if collected_samples == num_samples:
                break
                
        collected_means = True
        
    samples = torch.stack(samples)
    # print("total samples: ", samples.shape[0])
    mean = torch.stack(means).mean(dim=0)
    
    return mean, samples, None

=======
>>>>>>> origin/wmhuq
def ssn_mean_and_samples_3D(inputs):
    model_raw=inputs['model']
    x=inputs['x']
    y=inputs['y']
    num_samples = inputs['num_samples']
    remove_last=False
    if num_samples %2 != 0:
        num_samples +=1
        remove_last=True
    mean, sample = model_raw.mean_and_sample(x.cuda(), num_samples=num_samples)
    mean = mean[0].squeeze(0).cpu()
    sample = sample[:,0]
    # print(mean.shape)
    # print(sample.shape)
    
    if remove_last:
        sample = sample[:-1]

    return mean, sample, None


MODEL_OUTPUT_GENERATORS = {
    "deterministic":deterministic_mean,
    # "mc_drop":mc_drop_mean_and_samples,
    # "evidential":evid_mean,
    "ssn":ssn_mean_and_samples,
    # "punet":punet_mean_and_samples,
    # "ind":ssn_mean_and_samples,
    # "ens":ensemble_mean_and_samples,
    # "ssn_ens":ssn_ensemble_mean_and_samples,
}

def pad_to_original_space_2d(arr, cropped_dims, orig_dims, xs, ys):
    ndims = len(arr.shape)
    arr = crop_or_pad_dims(arr, (ndims-2, ndims-1), cropped_dims)
    shape = list(arr.shape)
    shape[-2] = orig_dims[0]
    shape[-1] = orig_dims[1]
    out = torch.zeros(shape, device=arr.device, dtype=arr.dtype)
    if len(shape) == 4:
        out[0] = 100 # so that where there is no image backgroud class = 1
        out[1:] = -100
        out[:, :, xs[0]:xs[1]+1, ys[0]:ys[1]+1] = arr
    elif len(shape) == 5:
        out[:, 0] = 100 # so that where there is no image backgroud class = 1
        out[:, 1:] = -100
        out[:, :, :, xs[0]:xs[1]+1, ys[0]:ys[1]+1] = arr
    else:
        raise ValueError(f"not defined for {len(shape)}D array")
    
    return out

def pad_to_original_space(arr, cropped_dims, orig_dims, zs, xs, ys):
    ndims = len(arr.shape)
    arr = crop_or_pad_dims(arr, (ndims-3, ndims-2, ndims-1), cropped_dims)
    shape = list(arr.shape)
    shape[-3] = orig_dims[0]
    shape[-2] = orig_dims[1]
    shape[-1] = orig_dims[2]
    out = torch.zeros(shape, device=arr.device, dtype=arr.dtype)
    if len(shape) == 4:
        out[0] = 100 # so that where there is no image backgroud class = 1
        out[1:] = -100
        out[:, zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1] = arr
    elif len(shape) == 5:
        out[:, 0] = 100 # so that where there is no image backgroud class = 1
        out[:, 1:] = -100
        out[:, :, zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1] = arr
    else:
        raise ValueError(f"not defined for {len(shape)}D array")
    
    return out

def get_nnunet_means_and_samples(model_raw, eval_ds, num_samples, model_func_name='deterministic', args={}):
    model_func = MODEL_OUTPUT_GENERATORS[model_func_name]
    means = []
    samples = []
    miscs = []

    model_raw.eval()
    for i, data in enumerate(tqdm(eval_ds, position=0, leave=True, ncols=100)):
        x = data[0]
        y = data[1]

        if y is not None:
            y = y.squeeze()
            y * (y==1).type(y.dtype)

        orig_spatial_dims = [x.shape[-3], x.shape[-2], x.shape[-1]]
        
        wheres = torch.where(x[-1])
        zs = (wheres[0].min().item(), wheres[0].max().item())
        xs = (wheres[1].min().item(), wheres[1].max().item())
        ys = (wheres[2].min().item(), wheres[2].max().item())
        # print(zs, xs, ys)
        # print(x.shape, y.shape)
    
        x = x[:, zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1]
        if y is not None:
            y = y[zs[0]:zs[1]+1, xs[0]:xs[1]+1, ys[0]:ys[1]+1]
    
        # print(x.shape, y.shape)

        cropped_spatial_dims = [x.shape[-3], x.shape[-2], x.shape[-1]]
        
        x = crop_or_pad_dims(x, [1,2,3], [48, 192, 192])
        if y is not None:
            y = crop_or_pad_dims(y, [0,1,2], [48, 192, 192])
    
        x = x.unsqueeze(0)
        
        
        inputs = {"model":model_raw, "x":x, "y":y, "num_samples":num_samples, "args":args}
        with torch.no_grad():
            mean, sample, misc = model_func(inputs)
            mean = pad_to_original_space(mean, cropped_spatial_dims, orig_spatial_dims, zs, xs, ys)
            means.append(mean.cpu())
            
            
            if sample != None:
                ndims = len(sample.shape)
                sample = pad_to_original_space(sample, cropped_spatial_dims, orig_spatial_dims, zs, xs, ys)
                sample = sample.cpu()
            samples.append(sample)
            miscs.append(misc)
            
    return means, samples, miscs

def get_means_and_samples(model_raw, eval_ds, num_samples, model_func=deterministic_mean, args={}):
    means = []
    samples = []
    miscs = []

    model_raw.eval()
    for i, data in enumerate(tqdm(eval_ds, position=0, leave=True, ncols=100)):
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

<<<<<<< HEAD

def resize_2d(x, y):
    # print(x.shape)
    
    orig_spatial_dims = [x.shape[-2], x.shape[-1]]
        
    wheres = torch.where(x[-1])
    xs = (wheres[1].min().item(), wheres[1].max().item())
    ys = (wheres[2].min().item(), wheres[2].max().item())
    # print(zs, xs, ys)
    # print(x.shape, y.shape)

    x = x[:, :, xs[0]:xs[1]+1, ys[0]:ys[1]+1]
    if y is not None:
        y = y[:, xs[0]:xs[1]+1, ys[0]:ys[1]+1]

    # print(x.shape, y.shape)

    cropped_spatial_dims = [x.shape[-2], x.shape[-1]]
    
    x = crop_or_pad_dims(x, [2,3], [192, 192])
    if y is not None:
        y = crop_or_pad_dims(y, [1,2], [192, 192])

    # print(x.shape)
    return x, y, orig_spatial_dims, cropped_spatial_dims, xs, ys

def get_means_and_samples_2D(model_raw, eval_ds, num_samples, model_func=deterministic_mean, args={}, resize=False):
=======
def get_means_and_samples_2D(model_raw, eval_ds, num_samples, model_func=deterministic_mean, args={}):
>>>>>>> origin/wmhuq
    means = []
    samples = []
    miscs = []

<<<<<<< HEAD
    if isinstance(model_raw, list):
        for m in model_raw:
            m.eval()
    else:
        model_raw.eval()
    for i, data in enumerate(tqdm(eval_ds, position=0, leave=True, ncols=100)):
        x = data[0]
        y = data[1]
        if resize:
            x, y, orig_spatial_dims, cropped_spatial_dims, x1s, x2s = resize_2d(x, y)
        
        inputs = {"model":model_raw, "x":x.squeeze().swapaxes(0, 1), "y":y.squeeze().swapaxes(0, 1) if y is not None else None, "num_samples":num_samples, "args":args}
        with torch.no_grad():
            mean, sample, misc = model_func(inputs)
            mean = mean.swapaxes(0, 1)
            if resize:
                mean = pad_to_original_space_2d(mean, cropped_spatial_dims, orig_spatial_dims, x1s, x2s)
            mean = mean.unsqueeze(0)
            means.append(mean.cpu())
            if sample != None:
                sample = reorder_samples(sample)
                sample = sample.swapaxes(1, 2)
                if resize:
                    sample = pad_to_original_space_2d(sample, cropped_spatial_dims, orig_spatial_dims, x1s, x2s)
                sample = sample.unsqueeze(1)
=======
    model_raw.eval()
    for i, data in enumerate(tqdm(eval_ds, position=0, leave=True, ncols=100)):
        x = data[0]
        y = data[1]
        inputs = {"model":model_raw, "x":x.squeeze().swapaxes(0, 1), "y":y.squeeze().swapaxes(0, 1), "num_samples":num_samples, "args":args}
        with torch.no_grad():
            mean, sample, misc = model_func(inputs)
            mean = mean.swapaxes(0, 1).unsqueeze(0)
            means.append(mean.cpu())
            if sample != None:
                sample = reorder_samples(sample)
                sample = sample.swapaxes(1, 2).unsqueeze(1)
>>>>>>> origin/wmhuq
                sample = sample.cpu()
                
            samples.append(sample)
            miscs.append(misc)
            
    return means, samples, miscs

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
    for idx in tqdm(range(len(means)), position=0, leave=True, ncols=100):
        umap_params = {"mean":means[idx], "samples":samples[idx], "misc":misc[idx], "do_normalize":args.uncertainty_type != "evidential"}
        ent_maps.append(umap_func(**umap_params))
    return ent_maps


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
    
