from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load best model
model = pickle.load(open("models/best_model.pkl", "rb"))

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    try:
        features = [float(x) for x in request.form.values()]
        final_input = [np.array(features)]

        prediction = model.predict(final_input)

        return render_template("index.html",
                               prediction_text=f"🌾 Recommended Crop: {prediction[0]}")

    except:
        return render_template("index.html",
                               prediction_text="⚠️ Invalid Input! Please check values.")

if __name__ == "__main__":
    app.run(debug=True)
