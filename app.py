from flask import Flask,request,render_template
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from src.pipeline.predict_pipeline import CustomData,PredictPipeline

app = Flask(__name__)


@app.route('/')
def home():
    # First load → no result shown
    return render_template('index.html', results=None)


@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():

    if request.method == 'GET':
        return render_template('index.html', results=None)

    try:
        data = CustomData(
            age=float(request.form.get('age')),
            anaemia=int(request.form.get('anaemia')),
            creatinine_phosphokinase=float(request.form.get('creatinine_phosphokinase')),
            diabetes=int(request.form.get('diabetes')),
            ejection_fraction=float(request.form.get('ejection_fraction')),
            high_blood_pressure=int(request.form.get('high_blood_pressure')),
            platelets=float(request.form.get('platelets')),
            serum_creatinine=float(request.form.get('serum_creatinine')),
            serum_sodium=float(request.form.get('serum_sodium')),
            sex=int(request.form.get('sex')),
            smoking=int(request.form.get('smoking')),
            time=float(request.form.get('time'))
        )

        pred_df = data.get_data_as_data_frame()

        predict_pipeline = PredictPipeline()
        result = predict_pipeline.predict(pred_df)

        return render_template('index.html', results=int(result[0]))

    except Exception as e:
        # If anything breaks, show clean page instead of crashing
        print(e)
        return render_template('index.html', results=None)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True)