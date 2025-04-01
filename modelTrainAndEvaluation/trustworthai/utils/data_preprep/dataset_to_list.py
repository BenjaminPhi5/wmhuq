def dataset_to_list(dataset, stride=1):
    output = dataset[0]
    elements = len(output)
    output_lists = [[] for _ in range(elements)]
    count = 0
    
    for output in dataset:
        if count % stride == 0:
            for i, value in enumerate(output):
                output_lists[i].append(value.squeeze())
        count += 1
            
    return output_lists