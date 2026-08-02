# ⚡ PJM Grid Energy Demand Forecast

> An interactive **Streamlit dashboard** that predicts **next-day electricity demand** for the **PJM Interconnection** using a **Gradient Boosting Regressor** trained on real historical grid demand data (2002–2018).

<p align="center">

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-1.8.0-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green?style=for-the-badge)

</p>

---

## 🌐 Live Demo

**🔗 Live App:** *Coming Soon*

---

## 📖 Overview

This project forecasts **next-day electricity demand** for the **PJM Interconnection**, one of North America's largest power grids.

Instead of relying on random train-test splits, the model follows proper **time-series forecasting practices**, using chronological validation and engineered historical demand features to predict future energy consumption.

The application is deployed through an interactive **Streamlit dashboard** that allows users to visualize historical demand, inspect model performance, and generate demand forecasts.

---

## ✨ Features

- 📈 Forecast next-day electricity demand
- ⚡ Gradient Boosting Regression model
- 📅 Calendar-based feature engineering
- 🔄 Lag-based historical demand features
- 📊 Actual vs Predicted visualization
- 🎯 Predicted vs Actual scatter plot
- 📌 Feature importance analysis
- 🧮 Baseline comparison ("Yesterday = Today")
- 🖥️ Interactive Streamlit dashboard
- 🚀 Instant predictions (no retraining required)

---

## 🧠 Model

### Features Used

**Calendar Features**

- Day of week
- Month
- Season
- Weekend indicator

**Historical Demand Features**

- Previous day's demand
- Same day last week's demand
- 7-day rolling average

### Validation Strategy

The model uses:

- ✅ Chronological Train/Test Split
- ✅ `TimeSeriesSplit` Cross Validation

This prevents **data leakage**, ensuring future observations never influence model training.

---

## 📊 Dashboard

The Streamlit dashboard includes:

- Historical demand visualization
- Actual vs Predicted comparison
- Prediction scatter plot
- Feature importance chart
- Interactive custom demand prediction

> All predictions are reported in **Megawatts (MW)** for the **PJM grid region**.

---

## 📂 Project Structure

```text
energy_app/
│
├── app.py
├── train_model.py
├── model.joblib
├── processed_history.csv
├── test_results.csv
├── feature_importance.csv
├── metrics.json
├── requirements.txt
└── requirements-train.txt
```

---

## 🚀 Running Locally

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 🔄 Retraining the Model

The training script downloads the latest dataset directly from **Kaggle** using **kagglehub**.

### Steps

1. Create a Kaggle account.
2. Generate an API token (`kaggle.json`).
3. Place it inside:

**Linux / macOS**

```text
~/.kaggle/kaggle.json
```

**Windows**

```text
C:\Users\<username>\.kaggle\kaggle.json
```

Alternatively, set:

```
KAGGLE_USERNAME
KAGGLE_KEY
```

Install training dependencies

```bash
pip install -r requirements-train.txt
```

Run training

```bash
python train_model.py
```

This regenerates:

- `model.joblib`
- `processed_history.csv`
- `test_results.csv`
- `feature_importance.csv`
- `metrics.json`

---

## ⚠️ Version Compatibility

The saved model was trained with

```
scikit-learn==1.8.0
```

Using a different version may produce serialization errors such as

```python
ModuleNotFoundError: No module named '_loss'
```

The deployment requirements are pinned to ensure compatibility.

---

## ☁️ Deployment

Deploy for free using **Streamlit Community Cloud**.

1. Push the project to GitHub
2. Create a new Streamlit app
3. Select your repository
4. Choose `app.py`
5. Deploy

No API keys or secrets are required because the trained model and processed datasets are already included in the repository.

---

## 📊 Dataset

**Source**

**Hourly Energy Consumption (PJM Interconnection)**

Originally published on Kaggle by **robikscube**.

https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption

---

## 🛠️ Tech Stack

| Category | Tools |
|-----------|-------|
| Language | Python |
| ML | Scikit-learn |
| Data | Pandas, NumPy |
| Visualization | Matplotlib |
| App | Streamlit |
| Model Storage | Joblib |

---

## 📌 Future Improvements

- Deep learning models (LSTM)
- XGBoost comparison
- Weather data integration
- Holiday-aware forecasting
- Multi-day forecasting
- Hyperparameter optimization

---

## 👨‍💻 Author

**Waleed Maqbool**

Electrical Engineering (Computer Engineering)  
FAST-NUCES Lahore

Interested in AI, Machine Learning, Predictive Analytics, and Intelligent Energy Systems.
