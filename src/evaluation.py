import numpy as np
import pandas as pd

def compute_coverage(y_pred_lo, y_pred_hi, y_test):
    coverage = np.sum((y_pred_lo <= y_test) * (y_test <= y_pred_hi)) / len(y_test)
    return coverage

def compute_average_width(y_pred_lo, y_pred_hi):
    return np.mean(y_pred_hi - y_pred_lo)

def evaluate_predictions(y_pred_lo, y_pred_hi, y_test, alpha, step_name=""):
    coverage = compute_coverage(y_pred_lo, y_pred_hi, y_test,)
    avg_width = compute_average_width(y_pred_lo, y_pred_hi)

    print(f"--- {step_name} ---")
    print(f"Coverage: {coverage:.4f} (Target: {1 - alpha})")
    print(f"Average Width: {avg_width:.4f} days\n")
    return coverage, avg_width

def evaluate_coverage_by_group(y_test, y_pred_lo, y_pred_hi, sensitive_feature):
    """
    Évalue la couverture et la largeur moyenne pour chaque sous-groupe (ex: Homme/Femme).
    sensitive_feature : tableau ou série Pandas contenant le genre pour chaque patient du test.
    """
    df_eval = pd.DataFrame({
        'vraie_duree': y_test.values if hasattr(y_test, 'values') else y_test,
        'borne_basse': y_pred_lo,
        'borne_haute': y_pred_hi,
        'groupe': sensitive_feature.values if hasattr(sensitive_feature, 'values') else sensitive_feature
    })

    print("\n--- Audit d'équité par groupe (Group-Balanced CP) ---")
    for groupe, data in df_eval.groupby('groupe'):
        couverture = np.mean((data['vraie_duree'] >= data['borne_basse']) & (data['vraie_duree'] <= data['borne_haute']))
        largeur_moy = np.mean(data['borne_haute'] - data['borne_basse'])
        
        print(f"Groupe '{groupe}' (N={len(data)}) :")
        print(f"  Couverture      : {couverture:.4f}")
        print(f"  Largeur moyenne : {largeur_moy:.2f} jours\n")