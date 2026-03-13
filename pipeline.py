# src/pipeline.py

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

from src.preprocessing import dataimport
import src.train as train 
import src.evaluate as evaluate
from src.models import RNN
from src.gpr_module import GPR

from src.utils import resolve_path, ensure_dir, setup_logging, create_sequences


def generate_paths(config):
    """
    Resolves all runtime paths from config. Supports both absolute and relative raw paths.
    gpr_dir is returned but not created here — it is only created inside the GPR branch.
    """
    dataset_info = config["dataset"]

    if os.path.isabs(dataset_info["file"]):
        raw_path = dataset_info["file"]
    else:
        raw_path = resolve_path(f"{config['paths']['raw']}/{dataset_info['file']}")

    forecasts_dir = resolve_path(f"{config['paths']['forecasts']}")
    log_dir = resolve_path(config['paths']['logs'])
    gpr_dir = resolve_path(config['paths']['gpr'])

    ensure_dir(forecasts_dir)
    ensure_dir(log_dir)
    ensure_dir(gpr_dir)

    return raw_path, forecasts_dir, log_dir, gpr_dir


def run_pipeline(dataset_name: str, config: dict, forecast_horizon, look_back, n_epochs, var, rnn_flag, output_folder=None):
    """
    Main pipeline. Handles data loading, scaling, model training, evaluation, and output saving.

    Args:
        dataset_name (str): key used for logging
        config (dict): full config dictionary
        forecast_horizon (int): number of future steps to predict (H)
        look_back (int): input rolling window length (L)
        n_epochs (int): training iterations (RNN only)
        var (str): target column name
        rnn_flag (int): 1 = RNN, 0 = GPR
        output_folder (str, optional): overrides config paths if provided
    Returns:
        final_rmse (float)
    """

    raw_path, forecasts_dir, log_dir, gpr_dir = generate_paths(config)

    # If the user specified an output folder from the GUI, override all config paths
    if output_folder is not None:
        output_folder = os.path.abspath(output_folder)
        ensure_dir(output_folder)
        forecasts_dir = os.path.join(output_folder, "forecasts")
        log_dir = os.path.join(output_folder, "logs")
        gpr_dir = os.path.join(output_folder, "gpr")

        ensure_dir(forecasts_dir)
        ensure_dir(log_dir)
        ensure_dir(gpr_dir)

    logger = setup_logging(dataset_name, log_dir)

    # Load and preprocess the target column
    df = pd.read_csv(raw_path, usecols=[var])
    df.columns = ['Value']
    df = dataimport(df)
    values = df['Value'].values.reshape(-1, 1)

    # Scale to [0, 1]
    scaler = MinMaxScaler(feature_range=(0, 1))
    values_scaled = scaler.fit_transform(values)

    # Temporal train/test split — last forecast_horizon steps held out as test
    train_data = values_scaled[:-forecast_horizon]
    test_data = values_scaled[-forecast_horizon:]

    ## -------------------------
    #  RECURRENT NEURAL NETWORK 
    ## -------------------------
    if rnn_flag == 1:
        n_neurons = config['model']['n_neurons']
        
        # Build supervised sequences from the training window
        x_train, y_train = create_sequences(train_data, look_back, forecast_horizon)
        x_train = np.reshape(x_train, (x_train.shape[0], x_train.shape[1], 1))

        rnn = RNN(x_train, y_train, n_epochs, forecast_horizon, n_neurons)

        # Callback runs evaluation and saves plots/CSV every n epochs
        test_callback = train.TestEveryNEpochs(
            n=config['test'].get('interval_epochs', 1),
            scaler=scaler,
            var=var,
            values=values,
            values_scaled=values_scaled,
            forecast_horizon=forecast_horizon,
            look_back=look_back,
            test_data=test_data,
            output_dir=forecasts_dir
        )

        model, history = train.train(
            model=rnn.RNN(),
            y_train=y_train,
            x_train=x_train,
            n_epochs=n_epochs,
            callbacks=[test_callback]
        )

        # Final forecast: take the last prediction from the test sequence
        x_test, y_test = create_sequences(values_scaled[:-forecast_horizon], look_back, forecast_horizon)
        x_test = np.reshape(x_test, (x_test.shape[0], x_test.shape[1], 1))
        forecast_scaled = model.predict(x_test)[-1]
        
        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))
        test_real = scaler.inverse_transform(test_data.reshape(-1, 1))

        final_rmse = np.sqrt(mean_squared_error(test_real, forecast))
        final_nrmse = final_rmse / np.mean(np.abs(test_real))
        with open(os.path.join(forecasts_dir, "rmse.txt"), "w") as f:
            f.write(f"RMSE: {final_rmse:.6f}\nNRMSE: {final_nrmse:.6f}\n")

        model_dir = os.path.join(forecasts_dir, "model")
        ensure_dir(model_dir)
        model.save(os.path.join(model_dir, "model.h5"))

        # Save epoch-by-epoch training loss
        csv_dir = os.path.join(forecasts_dir, "csv/training_loss")
        ensure_dir(csv_dir)
        df_loss = pd.DataFrame({
            "epoch": list(range(1, len(history.history["loss"]) + 1)),
            "loss": history.history["loss"]
        })
        df_loss.to_csv(os.path.join(csv_dir, "training_loss.csv"), index=False)

        return final_rmse
    
    else:

    ## --------------------------
    # GAUSSIAN PROCESS REGRESSION 
    ## --------------------------

        gpr_model = GPR(forecast_horizon)
        forecast_scaled = gpr_model.fit_and_predict(train_data)

        ensure_dir(gpr_dir)
        gpr_model.save(os.path.join(gpr_dir, "gpr_model.pkl"))

        forecast = scaler.inverse_transform(forecast_scaled.reshape(-1, 1))
        test_real = scaler.inverse_transform(test_data.reshape(-1, 1))

        final_rmse = np.sqrt(mean_squared_error(test_real, forecast))
        final_nrmse = final_rmse / np.mean(np.abs(test_real))

        with open(os.path.join(gpr_dir, "rmse.txt"), "w") as f:
            f.write(f"RMSE: {final_rmse:.6f}\nNRMSE: {final_nrmse:.6f}\n")

        forecast_df = pd.DataFrame({
            "true": test_real.flatten(),
            "forecast": forecast.flatten()
        })
        forecast_df.to_csv(os.path.join(gpr_dir, "gpr_forecast.csv"), index=False)

        # Plot full series: train context + test actuals + GPR forecast
        plt.figure(figsize=(10, 5))

        values_real = scaler.inverse_transform(values_scaled).flatten()
        n = len(values_real)
        split = n - forecast_horizon
        idx = range(n)

        plt.plot(idx[:split], values_real[:split], label="Reale (train)", color="blue")
        plt.plot(idx[split:], values_real[split:], label="Reale (test)", color="gray")
        plt.plot(idx[split:], forecast.flatten(), label="Forecast GPR", color="red")

        plt.axvline(split, linestyle="--", alpha=0.5)
        plt.title(f"GPR Forecast – RMSE = {final_rmse:.4f}  NRMSE = {final_nrmse:.4f}")
        plt.xlabel("Time step")
        plt.ylabel(var)
        plt.legend()
        plt.tight_layout()

        plt.savefig(os.path.join(gpr_dir, "gpr_forecast.png"))
        plt.close()

        return final_rmse