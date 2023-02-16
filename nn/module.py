from torch.nn import Sequential, Linear, BatchNorm1d, ReLU

def make_FC_encoder(input_size, encoder_size, batchnorm=True, activation=True, bias=True, BNmomentum=.5):
    """ Simple dense encoder.
    """
    layers = []
    for i, size in enumerate(encoder_size):
        layers.append(Linear(input_size if i == 0 else encoder_size[i-1], size, bias=bias))
        if batchnorm: layers.append(BatchNorm1d(size, momentum=BNmomentum))
        if activation: layers.append(ReLU())
    encoder = Sequential(*layers)
    return encoder

def make_FC_decoder(latent_size, decoder_size, batchnorm=True, activation=True, bias=True, BNmomentum=.5):
    """ Simple dense decoder.
    """
    layers = []
    for i, size in enumerate(decoder_size):
        layers.append(Linear(latent_size if i == 0 else decoder_size[i-1], size, bias=bias))
        if batchnorm: layers.append(BatchNorm1d(size, momentum=BNmomentum))
        if activation: layers.append(ReLU())
    decoder = Sequential(*layers)
    return decoder