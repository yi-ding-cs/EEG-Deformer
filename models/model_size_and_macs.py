from calflops import calculate_flops


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def get_parameters(model, input_shape):
    # input shape should be (1, cnn_chan, eeg_chan, time)
    flops, macs, params = calculate_flops(model=model,
                                          input_shape=input_shape,
                                          output_as_string=True,
                                          output_precision=4)
    print("FLOPs:%s   MACs:%s   Params:%s \n" % (flops, macs, params))