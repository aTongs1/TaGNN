import numpy as np

def R2(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        y_mean = np.mean(y_true)
        ss_total = np.sum((y_true - y_mean)**2)
        ss_residual = np.sum((y_true - y_pred)**2)
        r2 = 1 - (ss_residual / ss_total)
        r2 = np.nan_to_num(r2 * mask)
        r2 = np.mean(r2)
        return r2
    

def RMSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def RMSE_MAE_R2(y_true, y_pred):
    return (
        RMSE(y_true, y_pred),
        MAE(y_true, y_pred),
        R2(y_true, y_pred),
    )

