from flask import Flask, request, render_template, redirect, url_for
import numpy as np
import pandas as pd

from src.pipeline.predict_pipeline import CustomData, PredictPipeline

application = Flask(__name__)
app = application

# Redirect root URL to prediction page
@app.route('/')
def index():
    return redirect(url_for('predict_datapoint'))

# Route for prediction form and logic
@app.route('/predictdata', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            print("Form values received:", request.form.to_dict())

            data = CustomData(
                gender=request.form.get('gender'),
                race_ethnicity=request.form.get('ethnicity'),
                parental_level_of_education=request.form.get('parental_level_of_education'),
                lunch=request.form.get('lunch'),
                test_preparation_course=request.form.get('test_preparation_course'),
                writing_score=float(request.form.get('writing_score')),
                reading_score=float(request.form.get('reading_score'))
            )

            pred_df = data.get_data_as_data_frame()
            print("DataFrame passed to model:\n", pred_df)

            predict_pipeline = PredictPipeline()
            results = predict_pipeline.predict(pred_df)

            print("Prediction result:", results)

            return render_template('home.html', results=round(results[0], 2))

        except Exception as e:
            print("Exception during prediction:", e)
            return render_template('home.html', results="Error occurred: " + str(e))

# Run the Flask app
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
