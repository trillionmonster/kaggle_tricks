import tensorflow as tf
import math

def build_warmup_lr_function(
        lr_start=1e-8,
        lr_min=1e-8,
        lr_max=3e-5,
        lr_rampup_epochs=3,
        lr_sustain_epochs=0,
        n_cycles=.5,
        epochs=30,
):
    """
    
    :param lr_start: 
    :param lr_min: 
    :param lr_max: 
    :param lr_rampup_epochs: 
    :param lr_sustain_epochs: 
    :param n_cycles: 
    :param epochs: 
    :return: 
    """

    def lr_fn(epoch):

        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            progress = (epoch - lr_rampup_epochs - lr_sustain_epochs) / (epochs - lr_rampup_epochs - lr_sustain_epochs)
            lr = lr_max * (0.5 * (1.0 + tf.math.cos(math.pi * n_cycles * 2.0 * progress)))
            if lr_min is not None:
                lr = tf.math.maximum(lr_min, lr)

        return lr

    return lr_fn


