# OneAquaHealth dashboard 

This is a traditional Flask web application: a Python backend with a dedicated HTML, CSS, and JavaScript frontend.

## Start the app

1. Install Python 3.10 or later.
2. From this folder, install the required packages:

   ```powershell
   py -m pip install -r requirements.txt
   ```

3. Launch the traditional website:

   ```powershell
   py server.py
   ```

4. Open `http://127.0.0.1:5000` in a browser.

## Included experience

- AI water-sample classification using the five features required by `GEMStat_RandomForest_v2.pkl`
- Healthy / At-Risk probabilities and clear response guidance
- Persistent local log of community predictions in `GEMStat_Predictions.csv`
- Training-data ecosystem dashboard and model feature importance
- Citizen-science action flow and a responsible-use note

## File guide

| File or folder | Why it exists |
| --- | --- |
| `server.py` | The backend. It calls `predict_water_quality.py`, receives user measurements, and returns results to the website. |
| `templates/index.html` | The page structure and visible content: headings, form fields, dashboard sections, and placeholders for results. |
| `static/css/style.css` | The visual design: colors, layout, spacing, mobile responsiveness, and result-card styles. |
| `static/js/app.js` | Browser behavior: sends form data to the backend without refreshing the page and updates the prediction/dashboard. |
| `GEMStat_RandomForest_v2.pkl` | Trained Random Forest model loaded by `predict_water_quality.py`. |
| `predict_water_quality.py` | Loads `GEMStat_RandomForest_v2.pkl` from this folder and classifies a sample. |
| `train_model.py` | Trains the Random Forest on `GEMStat_RF_Final.csv` and saves `GEMStat_RandomForest_v2.pkl`. |
| `GEMStat_RF_Final.csv` | The reference GEMStat dataset used for training and dashboard statistics. |
| `GEMStat_Predictions.csv` | The local history of predictions made through the application. |
| `GEMStat_Test_Predictions_v2.csv` | Your saved hold-out predictions for the v2 model. |
| `requirements.txt` | The external Python packages that must be installed before the server can run. |
