import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

def evaluate_model(model, X_test, y_test, scaler):

    predictions = model.predict(X_test)

    # inverse transform
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test)

    print("\nLSTM Model Performance:")
    print("MSE:", mean_squared_error(y_test, predictions))
    print("MAE:", mean_absolute_error(y_test, predictions))

    # ✅ Safe baseline
    if len(y_test) > 1:
        baseline_pred = y_test[:-1]
        baseline_true = y_test[1:]

        baseline_mse = mean_squared_error(baseline_true, baseline_pred)

        print("\nBaseline Model (Naive):")
        print("MSE:", baseline_mse)
    else:
        print("\nNot enough data for baseline comparison")

    # 📊 Plot
    plt.figure()
    plt.plot(y_test, label="Actual")
    plt.plot(predictions, label="Predicted")
    plt.legend()
    plt.title("LSTM Forecasting")
    plt.show()