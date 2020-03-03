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
HP_CELLS_LSTM = hp.HParam("LSTM Cells", hp.Discrete([2, 4, 6, 8]))
HP_LR_LSTM = hp.HParam("learning_rate", hp.Discrete([0.0003]))
HP_IN_DROPOUT_LSTM = hp.HParam("Input Dropout", hp.Discrete([0.2, 0.4]))
HP_DROPOUT_LSTM = hp.HParam("Recurrent Dropout", hp.Discrete([0.0, 0.2, 0.4]))

# ConvLSTM hparams
HP_FILTERS_CL = hp.HParam("filters", hp.Discrete([8, 10]))
HP_KERNEL_CL = hp.HParam("kernel_size", hp.Discrete([5]))
HP_STRIDES_CL = hp.HParam("stride", hp.Discrete([1, 2, 3]))
HP_DROPOUT_CL = hp.HParam("dropout", hp.Discrete([0.4]))
HP_LSTMCELLS_CL = hp.HParam("LSTM Cells", hp.Discrete([4, 8, 12]))
HP_LR_CL = hp.HParam("learning_rate", hp.Discrete([0.0001, 0.0005]))

# ChannelCNN hparams
HP_CDEEP_CHANNELS = hp.HParam("Channel Filters", hp.Discrete([2, 3, 4, 5]))
HP_CDEEP_LAYERS = hp.HParam("Channel Kernel Size", hp.Discrete([1, 2, 3]))

# DeepCNN hparams
HP_DEEP_CHANNELS = hp.HParam("Filter start size", hp.Discrete([5, 10]))
HP_DEEP_LAYERS = hp.HParam("DownConv Layers", hp.Discrete([2, 3, 4]))
HP_DEEP_KERNEL_SIZE = hp.HParam("Kernel Size", hp.Discrete([3, 5]))
HP_LOSS_TYPE = hp.HParam("Loss Type", hp.Discrete(["BCE", "DUAL_BCE"]))

# LateFuseCNN hparams
HP_LDEEP_V_CHANNELS = hp.HParam("View-Ind. Filter start size", hp.Discrete([2, 4]))
HP_LDEEP_V_LAYERS = hp.HParam("View-Ind DownConv Layers", hp.Discrete([1, 2]))
HP_LDEEP_F_LAYERS = hp.HParam("Fused DownConv Layers", hp.Discrete([1, 2, 3]))
HP_LDEEP_DROPOUT = hp.HParam("dropout", hp.Discrete([0.4]))
HP_LDEEP_KSIZE = hp.HParam("kernel_size", hp.Discrete([3, 5, 10]))
HP_LDEEP_WEIGHTNORM = hp.HParam("Weight maxnorm constraint", hp.Discrete([2]))

OPT_PARAMS = {
    "BaseNET": {
        HP_FILTERS: 8,
        HP_DROPOUT: 0.4,
        HP_KERNEL: 5,
        HP_DILATION: 1,
        HP_POOL: 5
    },
    "DeepCNN": {
        HP_DEEP_LAYERS: 3,
        HP_DEEP_CHANNELS: 4,
        HP_DEEP_KERNEL_SIZE: 3,
        HP_LOSS_TYPE: "DUAL_BCE"
    }
}

METRIC_ACC = hp.Metric("accuracy", display_name="Accuracy")