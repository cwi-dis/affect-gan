from tensorflow.python.ops import math_ops
from collections import defaultdict

def binary_discretize_labels(labels, threshold=5.0): return math_ops.cast(labels > threshold, labels.dtype)


def _hparams_to_string(hparams):
    s = ""
    for k,v in hparams.items():
        s += "%s:%s, " % (k.name, v)

    return s[:-2]

def di():
    return defaultdict(int)

def dl():
    return defaultdict(list)