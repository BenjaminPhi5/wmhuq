import torch
from trustworthai.utils.logits_to_preds import normalize_samples
from trustworthai.utils.print_and_write_func import print_and_write

def VVC(v, normalize):
        if normalize:
            v = normalize_samples(v)
        return torch.std(v) / torch.mean(v)
    
def VVC_corr_coeff(results_text_file, ys3d, samples3d, tensor_alldice3d, do_normalize):
    print("this is a bad analysis that I should just stop using in this form")
    vvcs = [VVC(samples3d[i], do_normalize) for i in range(len(ys3d))]

    medians = torch.median(tensor_alldice3d, dim=0)[0]

    print_and_write(results_text_file, "vvc correlation coefficient:", newline=1)
    print_and_write(results_text_file, torch.corrcoef(torch.stack([torch.Tensor(vvcs), medians]))[0][1])