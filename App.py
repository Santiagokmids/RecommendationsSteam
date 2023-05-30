from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.clustering import load_model, predict_model
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model("kmeans_deployment")
cols = ["genre", "price_final", "win", "mac", "linux", "positive_ratio"]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features).reshape(1, -1)
    data_unseen = pd.DataFrame(final, columns = cols)
    prediction = predict_model(model, data=data_unseen)
    prediction = prediction.iloc[0]
    
    return render_template("home.html", pred = "El juegos recomendados es {}".format(prediction))

@app.route("/predict_api", methods = ["POST"])
def predict_api():
    data = request.get_json(force = True)
    data_unseen = pd.DataFrame([data])
    prediction = predict_model(model, data=data_unseen)
    output = prediction.iloc[0]
    
    return jsonify(output)

if __name__ == "__main__":
    app.run(debug=True)