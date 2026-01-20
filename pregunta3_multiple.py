# ===========================================
# PREGUNTA 3: REGRESIÓN MÚLTIPLE
# ===========================================
# Genera: generated/pregunta3.tex

import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs('generated', exist_ok=True)
FILE_NAME = 'nba_2010_2024_preprocesado_completo.csv'

# Cargar o generar datos
df = pd.read_csv(FILE_NAME)
  
# Predictores y variable dependiente
predictores = ['AST', 'REB', 'STL', 'TOV', 'FG_PCT']
X = df[predictores].dropna()
y = df.loc[X.index, 'PLUS_MINUS']

# Matriz de diseño con intercepto
X_mat = np.column_stack([np.ones(len(X)), X.values])

# Coeficientes usando pseudo-inversa para robustez
XtX_inv = np.linalg.pinv(X_mat.T @ X_mat)
beta = XtX_inv @ X_mat.T @ y.values

# Predicciones y residuos
y_pred = X_mat @ beta
residuos = y.values - y_pred
n_obs, k = len(y), len(predictores)

# Estadísticas R²
ss_res = np.sum(residuos**2)
ss_tot = np.sum((y.values - y.mean())**2)
r2 = 1 - ss_res/ss_tot
r2_adj = 1 - (1-r2)*(n_obs-1)/(n_obs-k-1)

# Error estándar de los coeficientes
mse = ss_res / (n_obs - k - 1)
se_beta = np.sqrt(np.diag(XtX_inv) * mse)

# Estadísticos t y p-values
t_stats = beta / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_obs - k - 1))

# Estadístico F global
f_stat = (r2 / k) / ((1 - r2) / (n_obs - k - 1))

# Mostrar resultados
print("\n===== Resultados de la Regresión Múltiple =====")
print(f"R² = {r2:.4f}")
print(f"R² ajustado = {r2_adj:.4f}")
print(f"F-statistic = {f_stat:.3f} (gl_modelo={k}, gl_residuo={n_obs-k-1})\n")

print("Coeficientes:")
print(f"{'Variable':<10} {'Beta':>8} {'SE':>8} {'t':>8} {'p-value':>10}")
print("-"*50)
print(f"{'Intercept':<10} {beta[0]:>8.3f} {se_beta[0]:>8.3f} {t_stats[0]:>8.2f} {p_values[0]:>10.3g}")
for i, var in enumerate(predictores):
    print(f"{var:<10} {beta[i+1]:>8.3f} {se_beta[i+1]:>8.3f} {t_stats[i+1]:>8.2f} {p_values[i+1]:>10.3g}")
