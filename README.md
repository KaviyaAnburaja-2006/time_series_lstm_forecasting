# 📊 Time Series Forecasting using LSTM

## 📌 Overview

This project implements a Deep Learning model using Long Short-Term Memory (LSTM) networks to perform time series forecasting. The model predicts future values (such as sales) based on historical data.

---

## 🎯 Objectives

* Perform data preprocessing on time series data
* Generate sequences using sliding window technique
* Build and train an LSTM model
* Evaluate performance using error metrics
* Compare results with a baseline model

---

## 🧠 Methodology

1. Data loading and preprocessing
2. Data normalization using MinMaxScaler
3. Sequence generation for supervised learning
4. Model building using LSTM layers
5. Training and validation
6. Evaluation using MSE and MAE
7. Baseline comparison (naive forecasting)

---

## ⚙️ Technologies Used

* Python
* TensorFlow / Keras
* NumPy
* Pandas
* Matplotlib
* Scikit-learn

---

## 📁 Project Structure

```
time_series_lstm_forecasting/
│── data/
│    └── sales.csv
│── preprocess.py
│── model.py
│── train.py
│── evaluate.py
│── main.py
│── requirements.txt
│── README.md
```

---

## ▶️ How to Run

1. Clone the repository:

```
git clone https://github.com/your-username/time_series_lstm_forecasting.git
```

2. Navigate to project folder:

```
cd time_series_lstm_forecasting
```

3. Install dependencies:

```
pip install -r requirements.txt
```

4. Run the project:

```
python main.py
```

---

## 📊 Output

* Training and validation loss
* Predicted vs Actual values graph
* Evaluation metrics:

  * Mean Squared Error (MSE)
  * Mean Absolute Error (MAE)

---

## 📈 Sample Output

(Add your screenshot here)

```
## 📈 Output Graph
![Output](screenshot.png)
```

---

## 🔮 Future Improvements

* Use larger real-world datasets
* Implement GRU / Transformer models
* Hyperparameter tuning
* Deploy using Streamlit or Flask

---

## 📌 Conclusion

The LSTM model effectively captures temporal dependencies in time series data and provides accurate forecasting results compared to baseline methods.

---

## 🙌 Author
Kaviya A

---
