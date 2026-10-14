import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from pathlib import Path

plot_dir = Path("data/plots")
plot_dir.mkdir(parents=True, exist_ok=True)

def plot_conformal_intervals(y_test, y_pred_lo, y_pred_hi, y_pred_mean, num_samples=100):
    """
    Affiche un échantillon d'intervalles de prédiction triés.
    """
    # On rassemble les données dans un DataFrame pour faciliter le tri
    df_plot = pd.DataFrame({
        'Vraie_LOS': y_test.values if isinstance(y_test, pd.Series) else y_test,
        'Pred_Mean': y_pred_mean,
        'Borne_Basse': y_pred_lo,
        'Borne_Haute': y_pred_hi
    })
    
    # On prend un échantillon aléatoire pour la lisibilité, puis on le trie par la prédiction moyenne
    df_sample = df_plot.sample(n=num_samples, random_state=42).sort_values(by='Pred_Mean').reset_index(drop=True)
    
    plt.figure(figsize=(12, 6))
    
    # Dessiner les barres d'intervalles
    plt.vlines(x=df_sample.index, ymin=df_sample['Borne_Basse'], ymax=df_sample['Borne_Haute'], 
               color='gray', alpha=0.5, label='Intervalle Conforme (90%)', linewidth=2)
    
    # Ajouter la prédiction moyenne
    plt.scatter(df_sample.index, df_sample['Pred_Mean'], color='blue', s=20, label='Moyenne Prédite (\u03bc\u0302)', zorder=3)
    
    # Ajouter les vraies valeurs
    plt.scatter(df_sample.index, df_sample['Vraie_LOS'], color='red', s=20, marker='x', label='Vraie Durée (Y)', zorder=4)

    plt.title(f"Échantillon de {num_samples} Prédictions Conformes (Triées par durée estimée)")
    plt.xlabel("Index Patient (Trié)")
    plt.ylabel("Durée de séjour (Jours)")
    plt.legend()
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(plot_dir / "conformal_intervals_sample.png", dpi=300)
    plt.show()

def plot_width_distribution(y_pred_lo, y_pred_hi, name=""):
    """
    Affiche la distribution des largeurs d'intervalles.
    """
    widths = y_pred_hi - y_pred_lo
    
    plt.figure(figsize=(10, 5))
    sns.countplot(x=widths, color="skyblue", edgecolor="black")
    
    plt.title("Distribution des largeurs d'intervalles")
    plt.xlabel("Largeur de l'intervalle (Jours)")
    plt.ylabel("Nombre de patients")
    plt.grid(True, axis='y', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    plt.savefig(plot_dir / f"{name}_interval_widths_distribution.png", dpi=300)
    plt.show()

def plot_group_fairness(y_test, y_pred_lo, y_pred_hi, sensitive_feature):
    """
    Génère un graphique à deux panneaux pour auditer l'équité de la prédiction conforme.
    """
    df_eval = pd.DataFrame({
        'vraie_duree': y_test.values if hasattr(y_test, 'values') else y_test,
        'borne_basse': y_pred_lo,
        'borne_haute': y_pred_hi,
        'groupe': sensitive_feature.values if hasattr(sensitive_feature, 'values') else sensitive_feature
    })
    
    results = []
    for groupe, data in df_eval.groupby('groupe'):
        cov = np.mean((data['vraie_duree'] >= data['borne_basse']) & (data['vraie_duree'] <= data['borne_haute']))
        width = np.mean(data['borne_haute'] - data['borne_basse'])
        results.append({'Groupe': groupe, 'Couverture': cov, 'Largeur_Moyenne': width})
        
    df_res = pd.DataFrame(results)
    
    # Création de la figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Graphique 1 : Couverture par groupe
    sns.barplot(data=df_res, x='Groupe', y='Couverture', ax=axes[0], palette=['#4C72B0', '#DD8452'])
    axes[0].axhline(y=0.90, color='red', linestyle='--', label='Cible (90%)')
    axes[0].set_ylim(0, 1.05)
    axes[0].set_title('Taux de Couverture par Genre')
    axes[0].set_ylabel('Couverture')
    axes[0].legend(loc='lower right')
    
    # Graphique 2 : Largeur moyenne par groupe
    sns.barplot(data=df_res, x='Groupe', y='Largeur_Moyenne', ax=axes[1], palette=['#4C72B0', '#DD8452'])
    axes[1].set_title("Largeur Moyenne de l'Intervalle")
    axes[1].set_ylabel('Jours')
    
    plt.tight_layout()
    plt.savefig(plot_dir / "grouped_conformal_prediction.png", dpi=300)
    plt.show()