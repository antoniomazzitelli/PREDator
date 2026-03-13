import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use("Agg")  # non-interactive backend — required for compiled/headless execution
import pandas as pd


def final_test(values_scaled, forecast_horizon, look_back, model, scaler, test_data):
    """
    Runs a single forecast from the last available training window and computes metrics.
    All metrics are computed on inverse-transformed (real-scale) values.

    Returns:
        test_real, forecast_real, rmse_test, rsquare, mae
    """

    # Extract the look_back window immediately before the test set
    input_seq = values_scaled[-forecast_horizon-look_back:-forecast_horizon] 
    input_seq = input_seq.reshape((1, look_back, 1)) 
    forecast_scaled = model.predict(input_seq, verbose=0) 
    forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1)) 
    test_real = scaler.inverse_transform(test_data)
    forecast_real = forecast.flatten()
    
    def RMSE(y_true, y_pred):
        return np.sqrt(np.mean((y_true - y_pred)**2))
    
    def r2(y_true, y_pred):
        ss_res = np.sum((y_true - y_pred)**2)
        ss_tot = np.sum((y_true - np.mean(y_true))**2)
        return 1 - ss_res / (ss_tot + 1e-10)
    
    def MAE(y_true, y_pred):
        return np.mean(np.abs(y_true - y_pred))
    
    rmse_test = RMSE(test_real.flatten(), forecast_real)
    rsquare   = r2(test_real.flatten(), forecast_real)
    mae       = MAE(test_real.flatten(), forecast_real)

    return test_real, forecast_real, rmse_test, rsquare, mae


def grafici(values, forecast_horizon, test_real, forecast_real, look_back, var, pathfig, pathcsv):
    """
    Plots recent history, test actuals, and RNN forecast. Saves figure and CSV to disk.
    """

    if len(values) < look_back + forecast_horizon:
        raise ValueError("look_back + forecast_horizon maggiore della lunghezza dei dati!")

    plt.figure(figsize=(12,6))
    
    x_range = np.arange(len(values) - forecast_horizon, len(values))
    
    plt.plot(
        range(len(values) - look_back - forecast_horizon, len(values) - forecast_horizon),
        values[-look_back - forecast_horizon:-forecast_horizon],
        label='Storico recente',
        color='gray', alpha=0.6
    )
    plt.plot(x_range, test_real, label='Dati reali (test)', color='blue')
    plt.plot(x_range, forecast_real, label='Previsione RNN', color='red', linestyle='--')
    
    plt.title('Confronto tra dati di test e previsione RNN')
    plt.xlabel('Datetime')
    plt.ylabel(var)
    plt.legend()
    plt.grid(True)
    
    plt.savefig(pathfig)
    plt.close()
    
    df = pd.DataFrame({"Datetime": x_range, "Forecast": forecast_real, "Ground Truth": test_real.flatten()})
    df.to_csv(pathcsv, index=False)
