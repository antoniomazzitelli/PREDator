import numpy as np
import matplotlib.pyplot as plt
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow.keras.callbacks as callback
import pandas as pd 
import numpy as np
import src.evaluate as evaluate


def train(model, y_train, x_train, n_epochs, batchsize=24, callbacks=None):
    """Fits the model and returns (model, history)."""
    history = model.fit(
        x_train,
        y_train,
        epochs=n_epochs,
        batch_size=batchsize,
        verbose=0,
        callbacks=callbacks
    )

    return model, history


from src.utils import ensure_dir


class TestEveryNEpochs(callback.Callback):
    """
    Keras callback that runs evaluation every n epochs and saves metrics, plots, and forecast CSVs.
    Output folders are created at instantiation time.
    """
    def __init__(self, n, scaler, var, values, values_scaled, forecast_horizon, look_back, test_data, output_dir):
        """
        Args:
            n (int): evaluation interval in epochs
            scaler: fitted MinMaxScaler used to inverse-transform predictions
            var (str): target variable name, used for file naming
            values (np.array): original (unscaled) time series
            values_scaled (np.array): scaled time series
            forecast_horizon (int)
            look_back (int)
            test_data (np.array): scaled test set
            output_dir (str): base output folder
        """
        super().__init__()
        self.n = n
        self.scaler = scaler
        self.var = var
        self.values = values
        self.values_scaled = values_scaled
        self.forecast_horizon = forecast_horizon
        self.look_back = look_back
        self.test_data = test_data
        self.output_dir = output_dir

        self.csv_dir = os.path.join(output_dir, "csv/test_loss")
        self.grafici_dir = os.path.join(output_dir, "grafici")
        self.pred_csv_dir = os.path.join(output_dir, "csv/predictions")

        ensure_dir(self.csv_dir)
        ensure_dir(self.grafici_dir)
        ensure_dir(self.pred_csv_dir)

    def on_epoch_end(self, epoch, logs=None):
        """Fires every n epochs — evaluates the model and appends metrics to test_loss.csv."""
        if (epoch + 1) % self.n == 0:
            test_real, forecast_real, rmse_test, rsquare, mae = evaluate.final_test(
                self.values_scaled,
                self.forecast_horizon,
                self.look_back,
                self.model,
                self.scaler,
                self.test_data
            )

            # Append metrics row — write header only on first epoch
            row = pd.DataFrame({"epoch": [epoch + 1], "rmse_test": [rmse_test], "mae": [mae], "rsquare": [rsquare]})
            csv_path = os.path.join(self.csv_dir, "test_loss.csv")
            if not os.path.exists(csv_path):
                row.to_csv(csv_path, index=False)
            else:
                row.to_csv(csv_path, index=False, mode='a', header=False)

            grafico_path = os.path.join(self.grafici_dir, f"{self.var}_epoch{epoch+1}.png")
            pred_csv_path = os.path.join(self.pred_csv_dir, f"{self.var}_epoch{epoch+1}.csv")

            evaluate.grafici(
                self.values,
                self.forecast_horizon,
                test_real,
                forecast_real,
                self.look_back,
                self.var,
                grafico_path,
                pred_csv_path
            )
