# src/gpr_module.py

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, RationalQuadratic as RQ, ConstantKernel as C, DotProduct as DP
from sklearn.metrics import mean_absolute_error
import joblib
import os


class GPR:
    """
    Gaussian Process Regression with greedy compositional kernel search.
    Kernel is selected by minimising MAE on a held-out validation tail.
    """

    def __init__(self, forecast_horizon=48, max_depth=5, val_size=24):
        self.forecast_horizon = forecast_horizon
        self.max_depth = max_depth
        self.val_size = val_size   # tail samples reserved for kernel validation

        self.model = None
        self.best_kernel = None
        self.x_train_len = None

    # ============================
    # BASE KERNELS
    # ============================
    @staticmethod
    def get_base_kernels():
        """Returns the three primitive kernels used as search building blocks."""
        return {
            "LIN": DP(),
            "RBF": RBF(),
            "RQ":  RQ()
        }

    @staticmethod
    def is_redundant(parent_name, new_name):
        return parent_name == new_name

    # ============================
    # EVALUATE KERNEL
    # ============================
    @staticmethod
    def evaluate_kernel(kernel, X_train, y_train, X_val, y_val):
        """Fits a GPR with the given kernel and returns MAE on the validation set. Returns inf on failure."""
        try:
            model = GaussianProcessRegressor(
                kernel=kernel,
                n_restarts_optimizer=10
            )
            model.fit(X_train, y_train)
            preds = model.predict(X_val)
            return mean_absolute_error(y_val, preds)
        except Exception:
            return np.inf

    # ============================
    # GREEDY SEARCH
    # ============================
    def kernel_search(self, train_data):
        """
        Greedy search over additive and multiplicative kernel combinations up to max_depth levels.
        At each level, expands the best kernel from the previous level with each base kernel.
        Sets self.best_kernel to the globally best candidate found.
        """

        y = np.array(train_data).reshape(-1)
        N = len(y)

        if N <= self.val_size:
            raise ValueError(f"val_size ({self.val_size}) must be smaller than training length ({N}).")

        # Temporal split — last val_size steps held out for kernel selection
        y_train_search = y[:-self.val_size]
        y_val = y[-self.val_size:]

        X_train_search = np.arange(len(y_train_search)).reshape(-1, 1)
        X_val = np.arange(len(y_train_search), N).reshape(-1, 1)

        base = self.get_base_kernels()
        candidates = [(name, kernel) for name, kernel in base.items()]
        global_best = (None, np.inf, None)

        for depth in range(self.max_depth):

            best_level = (None, np.inf, None)

            for name, kernel in candidates:
                mae = self.evaluate_kernel(
                    kernel,
                    X_train_search, y_train_search,
                    X_val, y_val
                )

                if mae < best_level[1]:
                    best_level = (name, mae, kernel)

                if mae < global_best[1]:
                    global_best = (name, mae, kernel)

            best_name, best_mae, best_kernel = best_level

            if best_kernel is None:
                break

            # Expand best kernel of this level via sum and product with each base kernel
            next_candidates = []

            for new_name, new_kernel in base.items():

                if not self.is_redundant(best_name, new_name):
                    next_candidates.append(
                        (f"({best_name} + {new_name})",
                         best_kernel + new_kernel)
                    )

                if not self.is_redundant(best_name, new_name):
                    next_candidates.append(
                        (f"({best_name} * {new_name})",
                         best_kernel * new_kernel)
                    )

            if len(next_candidates) == 0:
                break

            candidates = next_candidates

        self.best_kernel = global_best[2]
        return self.best_kernel

    # ============================
    # FIT + FORECAST
    # ============================
    def fit_and_predict(self, train_data, forecast_horizon=None):
        """
        Runs kernel search if needed, fits the GPR on the full training series,
        and returns the forecast array of length forecast_horizon.
        """

        forecast_horizon = forecast_horizon or self.forecast_horizon
        y_train = np.array(train_data).reshape(-1)
        self.x_train_len = len(y_train)

        # Run greedy search if no kernel has been set yet
        if self.best_kernel is None:
            self.kernel_search(y_train)

        X_full = np.arange(self.x_train_len).reshape(-1, 1)

        self.model = GaussianProcessRegressor(
            kernel=self.best_kernel,
            n_restarts_optimizer=20
        )
        self.model.fit(X_full, y_train)

        X_test = np.arange(
            self.x_train_len,
            self.x_train_len + forecast_horizon
        ).reshape(-1, 1)

        y_pred = self.model.predict(X_test)
        return y_pred

    # ============================
    # SAVE / LOAD
    # ============================
    def save(self, path):
        """Serializes model, kernel, and training length to disk via joblib."""
        dir_ = os.path.dirname(path)
        if dir_:
            os.makedirs(dir_, exist_ok=True)
        joblib.dump({
            "model": self.model,
            "kernel": self.best_kernel,
            "x_train_len": self.x_train_len
        }, path)

    @classmethod
    def load(cls, path):
        """Deserializes a saved GPR instance from disk."""
        data = joblib.load(path)
        obj = cls()
        obj.model = data["model"]
        obj.best_kernel = data["kernel"]
        obj.x_train_len = data["x_train_len"]
        return obj
