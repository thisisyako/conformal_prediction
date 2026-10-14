import numpy as np
import matplotlib.pyplot as plt


def get_cqr_confirmity_score_set(y_cal_lo, y_cal_hi, y_calib):
    return np.maximum(y_cal_lo - y_calib, y_calib - y_cal_hi)

def regression_conformity_score(f_xi, y_calib):
    return np.absolute(y_calib - f_xi)

def calibrate_cqr(y_cal_lo, y_cal_hi, y_calib, alpha):
    cal_score_set = get_cqr_confirmity_score_set(y_cal_lo, y_cal_hi, y_calib)
    n = len(y_calib)

    # On prend le quantile qu'on veut
    q_level = ((n + 1) * (1 - alpha)) / n
                      
    # On prend la vraie valeur dans notre ensemble de calibration
    # qui correspond au quantile `q_level`
    q_hat = np.quantile(cal_score_set, q_level, method="higher")

    return q_hat

def calibrate_scp(mean_model, error_model, X_calib, y_calibrate, alpha):

    # On récupère le score de conformité sur nos données de calibration
    X_calib_pred = mean_model.predict(X_calib)
    base_conf_score = regression_conformity_score(X_calib_pred, y_calibrate)

    # On calcule nos u(X) qui sont les erreurs préditent par notre modèle hgbr_error
    # pour chaque valeur de X_calib_pred
    u = error_model.predict(X_calib)
    u = np.maximum(u, 1e-6)

    # On récupère notre ensemble de scores de conformité normalisés
    calib_conf_score = base_conf_score / u

    # On prend le quantile qu'on veut
    n = len(y_calibrate)
    q_level = ((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)

    # On prend la vraie valeur dans notre ensemble de calibration
    # qui correspond au quantile `q_level`
    q_hat = np.quantile(calib_conf_score, q_level, method="higher")

    return q_hat

def predict_normalized(mean, error, X, q_hat):
    y_mean = mean.predict(X)
    u_pred = error.predict(X)

    lo_real, hi_real = y_mean - (q_hat * u_pred), y_mean + (q_hat * u_pred)

    # On discrétise nos bornes pour avoir des jours entiers
    lo_discrete = np.maximum(1, np.floor(lo_real))
    hi_discrete = np.ceil(hi_real)
    return lo_discrete, hi_discrete
