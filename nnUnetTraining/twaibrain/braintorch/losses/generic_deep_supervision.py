import torch
import torch.nn.functional as F

class DeepSupervisionLoss():
    def __init__(self, criterion, print_components=False):
        self.criterion = criterion
        self.print_components = print_components

    def __call__(self, outputs, target):
        main_output = outputs[0]
        deep_outputs = outputs[1:]
        loss = self.criterion(main_output, target.squeeze(dim=1).type(torch.long))
        if self.print_components:
            print(loss.item())
        
        # Apply deep supervision
        deep_supervision_weight = 0.5
        for deep_output in deep_outputs:
            # Resize target to match deep output size
            resized_target = F.interpolate(target.float(), size=deep_output.shape[2:], mode='nearest')
            loss_component = self.criterion(deep_output, resized_target.squeeze(dim=1).type(torch.long))
            if self.print_components:
                print(loss_component.item())
            loss += deep_supervision_weight * loss_component
            deep_supervision_weight *= deep_supervision_weight
    
        return loss

class MultiDeepSupervisionLoss():
    def __init__(self, criterions, print_components=False):
        self.criterions = criterions
        self.print_components = print_components

    def __call__(self, outputs, target):
        main_output = outputs[0]
        deep_outputs = outputs[1:]
        loss = self.criterions[0](main_output, target)
        if self.print_components:
            print(loss.item())
            
        # Apply deep supervision
        deep_supervision_weight = 0.5
        for deep_output, criterion in zip(deep_outputs, self.criterions[1:]):
            # Resize target to match deep output size
            resized_target = F.interpolate(target.float(), size=deep_output.shape[2:], mode='nearest')
            loss_component = criterion(deep_output, resized_target.squeeze(dim=1).type(torch.long))
            if self.print_components:
                print(loss_component.item())
            loss += deep_supervision_weight * loss_component
            deep_supervision_weight *= deep_supervision_weight
    
        return loss
