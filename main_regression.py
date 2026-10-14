
import argparse
import numpy as np

from pathlib import Path

from src.regression.conformal_regression import calibrate_cqr, calibrate_scp, predict_normalized
from src.regression.data_loader import create_data
from src.regression.evaluation import evaluate_predictions, evaluate_coverage_by_group
from src.regression.models import train_quantile_regressor, train_hgbr, get_or_train_qr, save_model
from src.regression.visualization import plot_conformal_intervals, plot_width_distribution, plot_group_fairness

ALPHA = 0.1

LO_MODEL_NAME = f"qr_lo-{ALPHA}.joblib"
HI_MODEL_NAME = f"qr_hi-{ALPHA}.joblib"

MEAN_MODEL_NAME = f"mean_model-{ALPHA}.joblib"
ERROR_MODEL_NAME = f"error_model-{ALPHA}.joblib"

model_dir = Path("data/models")
model_dir.mkdir(parents=True, exist_ok=True)

def cqr(grouped=False):
    X_train, X_calibrate, X_test, y_train, y_calibrate, y_test, _, gender_test = create_data()

    QR_lo, QR_hi = get_or_train_qr(
        model_dir / LO_MODEL_NAME,
        model_dir / HI_MODEL_NAME,
        X_train,
        y_train,
        ALPHA
    )

    # ----------------------------------
    # PHASE 1 : Base Quantile Regression
    # ----------------------------------

    y_pred_lo_base = QR_lo.predict(X_test)
    y_pred_hi_base = QR_hi.predict(X_test)
    
    _, _ = evaluate_predictions(
        y_pred_lo_base,
        y_pred_hi_base,
        y_test,
        alpha=ALPHA,
        step_name="1. Base Quantile Regression"
    )
    
    plot_width_distribution(y_pred_lo_base, y_pred_hi_base, "QR")

    # la LOS est majoritairement entre 1 et 4 jours donc l'objectif va être de réduire la largeur moyenne avec la CQR.

    # ---------------------------------------
    # PHASE 2 : Conformal Quantile Regression
    # ---------------------------------------

    y_cal_lo = QR_lo.predict(X_calibrate)
    y_cal_hi = QR_hi.predict(X_calibrate)

    q_hat = calibrate_cqr(y_cal_lo, y_cal_hi, y_calibrate, alpha=ALPHA)
    y_pred_lo_cqr = y_pred_lo_base - q_hat
    y_pred_hi_cqr = y_pred_hi_base + q_hat
    
    _, _ = evaluate_predictions(
        y_pred_lo=y_pred_lo_cqr,
        y_pred_hi=y_pred_hi_cqr,
        y_test=y_test,
        alpha=ALPHA,
        step_name="2. Conformal Quantile Regression"
    )

    if grouped:
        evaluate_coverage_by_group(y_test, y_pred_lo_cqr, y_pred_hi_cqr, gender_test)
        plot_group_fairness(y_test, y_pred_lo_cqr, y_pred_hi_cqr, gender_test)

def normalized_conformal_prediction(grouped=False):
    """"
    Le but ici est de faire une prediction conforme en entrainant un premier modele
    qui aura pour but de predire la moyenne de la duree du sejour d'un patient.
    Ensuite, on aura un modele d'incertitude, le u(x) qu'on a vu en cours. Son
    but est de predire l'erreur absolue d'entrainement du premier modele.
    Une fois qu'on a ca, on vient calibrer nos donnees, et on retrouve la valeur du
    quantile donné. On pourra ainsi prédir un intervalle continu en discrétisant la borne
    basse et haute.
    """
    X_train, X_calibrate, X_test, y_train, y_calibrate, y_test, _, gender_test = create_data()

    # On entraîne notre modèle qui doit prédire la moyenne de séjour d'un patient
    hgbr_mean = train_hgbr(X_train, y_train)

    # On prédit les valeurs des données de X_train
    X_pred = hgbr_mean.predict(X_train)
    error = np.absolute(y_train - X_pred)

    # On entraîne notre modèle qui doit prédire l'erreur du modèle hgbr_mean
    hgbr_error = train_hgbr(X_train, error)

    # On calibre nos données et on récupère la valeur au quantile voulu
    q_hat = calibrate_scp(hgbr_mean, hgbr_error, X_calibrate, y_calibrate, ALPHA)

    # On peut maintenant prédire avec le q_hat et nos 2 modèles.
    lo, hi = predict_normalized(hgbr_mean, hgbr_error, X_test, q_hat)

    evaluate_predictions(lo, hi, y_test, ALPHA, "Normalized Conformal Prediction")

    y_mean = hgbr_mean.predict(X_test)
    plot_conformal_intervals(y_test, lo, hi, y_mean, num_samples=100)
    plot_width_distribution(lo, hi, name=f"normalized_{ALPHA}")

    if grouped:
        evaluate_coverage_by_group(y_test, lo, hi, gender_test)
        plot_group_fairness(y_test, lo, hi, gender_test)

    save_model(hgbr_mean, model_dir / MEAN_MODEL_NAME)
    save_model(hgbr_error, model_dir / ERROR_MODEL_NAME)


if __name__ == "__main__" :
    parser = argparse.ArgumentParser(description="Pipeline de Prédiction Conforme pour la régression du LOS.")
    
    parser.add_argument(
        "--method", 
        type=str, 
        choices=["cqr", "normalized"], 
        default="normalized",
        help="Méthode de prédiction conforme à utiliser : 'cqr' ou 'normalized'."
    )
    
    parser.add_argument(
        "--grouped", 
        action="store_true", 
        help="Active l'évaluation de l'équité par groupe (Group-Balanced CP)."
    )

    args = parser.parse_args()

    # Exécution selon les arguments
    if args.method == "cqr":
        print("--- Exécution : Conformal Quantile Regression (CQR) ---")
        cqr(grouped=args.grouped)
        
    elif args.method == "normalized":
        print("--- Exécution : Normalized Conformal Prediction (SCP) ---")
        normalized_conformal_prediction(grouped=args.grouped)