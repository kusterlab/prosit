import os

from . import io_local
from . import losses
from . import model as model_lib
from . import constants


def get_callbacks(model_dir_path):
    import keras

    loss_format = "{val_loss:.5f}"
    epoch_format = "{epoch:02d}"
    weights_file = "{}/weight_{}_{}.hdf5".format(
        model_dir_path, epoch_format, loss_format
    )
    save = keras.callbacks.ModelCheckpoint(weights_file, save_best_only=True)
    stop = keras.callbacks.EarlyStopping(patience=10)
    decay = keras.callbacks.ReduceLROnPlateau(patience=2, factor=0.2)
    return [save, stop, decay]


def train(tensor, model, model_config, callbacks):
    import keras

    if isinstance(model_config["loss"], list):
        loss = [losses.get(l) for l in model_config["loss"]]
    else:
        loss = losses.get(model_config["loss"])
    optimizer = model_config["optimizer"]
    x = io_local.get_array(tensor, model_config["x"])
    y = io_local.get_array(tensor, model_config["y"])
    model.compile(optimizer=optimizer, loss=loss)
    model.fit(
        x=x,
        y=y,
        epochs=constants.TRAIN_EPOCHS,
        batch_size=constants.TRAIN_BATCH_SIZE,
        validation_split=1 - constants.VAL_SPLIT,
        callbacks=callbacks,
    )
    keras.backend.get_session().close()


if __name__ == "__main__":
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"  # turn off tf logging
    data_path = constants.DATA_PATH
    model_dir = constants.MODEL_DIR

    model, model_config = model_lib.load(model_dir, trained=True)
    tensor = io_local.from_hdf5(data_path)
    callbacks = get_callbacks(model_dir)
    train(tensor, model, model_config, callbacks)
