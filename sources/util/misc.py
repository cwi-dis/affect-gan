import tensorflow as tf
from tensorflow.python.ops import math_ops
from collections import defaultdict

def binary_discretize_labels(labels, threshold=5.0): return math_ops.cast(labels > threshold, labels.dtype)

def init_tf_gpus():
    print(tf.__version__)
    gpus = tf.config.experimental.list_physical_devices('GPU')
    print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            logical_gpus = tf.config.experimental.list_logical_devices('GPU')
            print("Physical GPUs available: %d  -- Logical GPUs available: %d" % (len(gpus), len(logical_gpus)))
        except RuntimeError as e:
            print(e)

def _hparams_to_string(hparams):
    s = ""
    for k,v in hparams.items():
        s += "%s:%s, " % (k.name, v)

    return s[:-2]

def di():
    return defaultdict(int)

def dl():
    return defaultdict(list)
