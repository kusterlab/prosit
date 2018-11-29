import os
import yaml

from . import constants
from . import layers
from . import utils


MODEL_NAME = "model.yml"
CONFIG_NAME = "config.yml"


def is_weight_name(w):
    return w.startswith("weight_") and w.endswith(".hdf5")


def get_loss(x):
    return float(x.split("_")[-1][:-5])


def get_best_weights_path(model_dir):
    weights = list(filter(is_weight_name, os.listdir(model_dir)))
    if len(weights) == 0:
        return None
    else:
        d = {get_loss(w): w for w in weights}
        weights_path = "{}/{}".format(model_dir, d[min(d)])
        return weights_path


def load(model_dir, trained=False):
    import keras

    model_path = os.path.join(model_dir, MODEL_NAME)
    config_path = os.path.join(model_dir, CONFIG_NAME)
    weights_path = get_best_weights_path(model_dir)
    with open(config_path, "r") as f:
        config = yaml.load(f)
    with open(model_path, "r") as f:
        model = keras.models.model_from_yaml(
            f.read(), custom_objects={"Attention": layers.Attention}
        )
    if trained and weights_path is not None:
        model.load_weights(weights_path)
    return model, config


def save(model, config, model_dir):
    model_path = MODEL_NAME.format(model_dir)
    config_path = CONFIG_NAME.format(model_dir)
    utils.check_mandatory_keys(config, ["name", "optimizer", "loss", "x", "y"])
    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
    with open(model_path, "w") as f:
        f.write(model.to_yaml())
