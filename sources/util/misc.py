def _hparams_to_string(hparams):
    s = ""
    for k,v in hparams.items():
        s += "%s:%s, " % (k.name, v)

    return s[:-2]