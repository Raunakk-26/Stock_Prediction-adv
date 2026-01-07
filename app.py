from flask import Flask, request, jsonify, render_template
from flask_cors import CORS
from ml_model import predict_future_prices

app = Flask(__name__)   # <-- DO NOT change this
CORS(app)


# ---------------------------
# ROUTES FOR WEB PAGES
# ---------------------------

@app.route("/")
def home():
    return render_template("index.html")


@app.route("/stock-prediction")
def stock_prediction():
    return render_template("stock_prediction.html")


# ---------------------------
# API ROUTE FOR ML PREDICTION
# ---------------------------

@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json()

    symbol = data.get("symbol")
    start_date = data.get("start_date")
    end_date = data.get("end_date")

    if not symbol or not start_date or not end_date:
        return jsonify({"error": "Missing input parameters"}), 400

    try:
        dates, prices = predict_future_prices(
            symbol=symbol,
            start_date=start_date,
            end_date=end_date
        )

        return jsonify({
            "dates": dates,
            "predicted_prices": prices
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ---------------------------
# RUN FLASK APP
# ---------------------------

if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
