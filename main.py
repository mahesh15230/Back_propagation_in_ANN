if __name__ == "__main__":
    data = 'dataset_full.txt'
    fulldata = data_init(data, full = 1)
    testdata = data_init(data, test = 1)
    traindata = data_init(data, train = 1)   
    validdata = data_init(data, valid = 1)

    minibatchsize = 64
    ann_arch = [4,9,15,5,1]
    layers = layers_init(ann_arch, minibatchsize)
    weights = weights_init(ann_arch)
    act_func = activations(ann_arch)



