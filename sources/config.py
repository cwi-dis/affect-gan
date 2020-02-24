from tensorboard.plugins.hparams import api as hp

INPUT_SIZE = 500

# BaseNET hparams
HP_FILTERS = hp.HParam("filters", hp.Discrete([2, 4, 8]))
HP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.5, 0.75]))
HP_KERNEL = hp.HParam("kernel_size", hp.Discrete([3, 5]))
HP_DILATION = hp.HParam("dilation", hp.Discrete([1, 3]))
HP_POOL = hp.HParam("pool_size", hp.Discrete([5, 10]))
HP_LR = hp.HParam("learning_rate", hp.Discrete([0.001, 0.0005]))

# SimpleLSTM hparams
HP_CELLS_LSTM = hp.HParam("LSTM Cells", hp.Discrete([4, 8, 12, 16]))
HP_LR_LSTM = hp.HParam("learning_rate", hp.Discrete([0.001, 0.0001]))

METRIC_ACC = hp.Metric("accuracy", display_name="Accuracy")