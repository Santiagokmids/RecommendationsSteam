from flask import Flask, request, url_for, redirect, render_template, jsonify
from pycaret.clustering import load_model, predict_model
import pandas as pd
import pickle
import numpy as np

app = Flask(__name__)

model = load_model("kmeans_deployment")
cols = ["genre", "price_final", "win", "mac", "linux", "positive_ratio","app_id"]

@app.route("/")
def home():
    return render_template("home.html")

@app.route("/predict", methods=["POST"])
def predict():
    int_features = [x for x in request.form.values()]
    final = np.array(int_features).reshape(1, -1)
    data_unseen = pd.DataFrame(final, columns = cols)
    prediction = predict_model(model, data=data_unseen)

    genre = prediction["genre"].values[0]
    price = int(prediction["price_final"].values[0])
    win = int(prediction["win"].values[0])
    mac = int(prediction["mac"].values[0])
    linux = int(prediction["linux"].values[0])

    values = searchGames(genre, price, win, mac, linux)
    arr = values["title"]

    if(arr.size > 0):
        return render_template("home.html", pred = "Los juegos recomendados son {}".format(", ".join(arr)))
    
    else:
        return render_template("home.html", pred = "Lo sentimos, no se pueden recomendar juegos con esas caracterísitcas :(")


def searchGames(genre, price, win, mac, linux):

    df = pd.read_csv("dataframe.csv")

    results = df.loc[(df['genre'] == genre) & (df['price_final'] >= price - 10000) & (df['price_final'] <= price + 10000)
                     & (df['win'] == win) & (df['mac'] == mac) & (df['linux'] == linux)].head(5)

    games = pd.DataFrame(results) 

    return games

if __name__ == "__main__":
    app.run(debug=True)