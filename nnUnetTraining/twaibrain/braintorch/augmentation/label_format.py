from torch import stack

class OneHotEncoder():
    """
    Convert labels to multi channels, one channel per class,
    including background (so output channel 0 is background class, 1 is WMH, 2 is other_pathology
    for the WMH channel dataset for example
    """
    def __init__(self, num_classes:int, key:str):
        """
        num_classes: number of classes in total (including any background class)
        key: the key in the data corresponding to the labels to be modified
        """
        super().__init__()
        self.num_classes = num_classes
        self.key = key

    def __call__(self, data):
        d = dict(data)
        labels = d[self.key]
        dtype = labels.dtype

        result = []
        for c in range(self.num_classes):
            result.append((labels[0] == c).type(dtype))
            
        if labels.shape[0] > 1:
            assert labels.shape[0] == 2 # assume that there is one extra channel for the mask....
            result.append(labels[1])

        d[self.key] = stack(result, axis=0)

        return d

class OneVRest():
    """
    Converts the segmentation task to a binary segmentation task
    by selecting for only one of the classes
    """
    def __init__(self, selected_class:int, key:str):
        """
        selected_class: the id of the class to select for
        key: the key in the data corresponding to the labels to be modified
        """
        super().__init__()
        self.selected_class = selected_class
        self.key = key

    def __call__(self, data):
        d = dict(data)
        labels = d[self.key]

        d[self.key] = (labels==self.selected_class).type(labels.dtype)

        return d
