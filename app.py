# app.py
from flask import Flask, render_template, request
import numpy as np
import pickle

app = Flask(__name__)

# Load saved artifact (feature names, scaler, models, target_names)
artifact = pickle.load(open("artifact_10feat.pkl", "rb"))
FEATURE_NAMES = artifact["feature_names"]     # list of 10 feature names
scaler = artifact["scaler"]
models = artifact["models"]                   # dict: 'lr','dt','rf'
target_names = artifact["target_names"]       # e.g. ['malignant','benign']

def human_label(pred):
    # pred is 0 or 1
    # target_names[0] -> malignant, [1] -> benign
    return target_names[pred].capitalize()  # "Malignant" or "Benign"

@app.route("/")
def home():
    # pass feature names to template so labels are meaningful
    return render_template("index.html", feature_names=FEATURE_NAMES)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Collect 10 values in the same order as FEATURE_NAMES
        values = []
        for i in range(len(FEATURE_NAMES)):
            key = f"f{i+1}"           # form names are f1..f10
            v = request.form.get(key, None)
            if v is None or v.strip() == "":
                return render_template("result.html", error="Please enter all feature values.")
            values.append(float(v))

        arr = np.array(values).reshape(1, -1)     # shape (1,10)

        # Scale using scaler fitted in training
        arr_scaled = scaler.transform(arr)

        # Individual predictions and probabilities
        preds = {}
        probs = {}
        for name, model in models.items():
            p = int(model.predict(arr_scaled)[0])     # 0 or 1
            preds[name] = p
            # if model has predict_proba:
            if hasattr(model, "predict_proba"):
                probs[name] = float(model.predict_proba(arr_scaled)[0, int(p)])
            else:
                probs[name] = None

        # Ensemble majority vote (simple majority)
        votes = preds["lr"] + preds["dt"] + preds["rf"]   # sum of 0/1
        final = 1 if votes >= 2 else 0

        # Prepare human readable outputs
        individual_readable = {k: human_label(v) for k, v in preds.items()}
        final_readable = human_label(final)

        return render_template(
            "result.html",
            feature_names=FEATURE_NAMES,
            input_values=values,
            individual_preds=individual_readable,
            individual_probs=probs,
            final_pred=final_readable,
            final_color=("red" if final_readable.lower()=="malignant" else "green")
        )

    except Exception as e:
        return render_template("result.html", error=str(e))


if __name__ == "__main__":
    app.run(debug=True)
