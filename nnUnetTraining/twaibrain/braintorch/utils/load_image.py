import SimpleITK as sitk
import torch
from twaibrain.braintorch.utils.resample import torch_resample

def torch_load(imgpath):
    return torch.from_numpy(sitk.GetArrayFromImage(sitk.ReadImage(imgpath))).type(torch.float32).unsqueeze(0)

def torch_load_and_resample(imgpath, out_spacing, orig_spacing=None, is_label=False):
    img = sitk.ReadImage(imgpath)
    if out_spacing is None:
        raise ValueError("out spacing must be defined")
    if orig_spacing is None:
        orig_spacing = img.GetSpacing()

    img = torch.from_numpy(sitk.GetArrayFromImage(img)).type(torch.float32).unsqueeze(0)

    img = torch_resample(img, out_spacing, orig_spacing, is_label=is_label)

    return img
