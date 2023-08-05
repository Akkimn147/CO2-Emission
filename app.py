import numpy as np
from flask import Flask, request, json, render_template
import pickle
from sklearn.preprocessing import StandardScaler


# Create Flask app
app = Flask(__name__)

# Load the pickle model

model = pickle.load(open("Co2model.pkl", "rb"))

scaler = StandardScaler()

@app.route("/")
def home():
    return render_template("mine.html")


@app.route("/predict", methods=["POST"])
def predict():

    float_features = [float(x) for x in request.form.values()]
    features = [np.array(float_features)]
    data = scaler.fit_transform(features)
    prediction = model.predict(data)
    return render_template("mine.html", prediction_text="Emission is:- {}".format(prediction))


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080)

# if __name__ == "__main__":
#     app.run(port=8080, debug=True)
