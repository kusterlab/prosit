import os
import numpy
import flask
import pandas


from . import model as model_lib
from . import io
from . import constants
from . import tensorize
from . import prediction
from . import alignment
from . import maxquant


app = flask.Flask(__name__)


@app.route("/")
def hello():
    return "prosit!\n"


@app.route("/predict/", methods=["POST"])
def predict():
    df = pandas.read_csv(flask.request.files["peptides"])
    tensor = tensorize.peptidelist(df)
    result = prediction.predict(tensor, model, model_config)
    df_pred = maxquant.convert_prediction(result)
    path = "{}prediction.csv".format(model_dir)
    maxquant.write(df_pred, path)
    return flask.send_file(path)


if __name__ == "__main__":
    model_dir = constants.MODEL_DIR
    global model
    global model_config
    model, model_config = model_lib.load(model_dir, trained=True)
    app.run(host="0.0.0.0")
