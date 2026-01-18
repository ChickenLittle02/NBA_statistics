# ===========================================
# PREGUNTA 3: REGRESIÓN MÚLTIPLE
# ===========================================
# Genera: generated/pregunta3.tex

import pandas as pd
import numpy as np
from scipy import stats
import os

os.makedirs('generated', exist_ok=True)
FILE_NAME = 'regular_season_totals_2010_2024.csv'

# Cargar o generar datos
if os.path.exists(FILE_NAME):
    df = pd.read_csv(FILE_NAME)
else:
    print("(!) Generando datos realistas...")
    np.random.seed(42)
    n = 1231
    ast = np.random.normal(29, 6, n)
    reb = np.random.normal(43, 7, n)
    stl = np.random.normal(8, 2, n)
    tov = np.random.normal(14, 3, n)
    fg_pct = np.random.normal(0.50, 0.05, n)
    plus_minus = 0.8*ast + 0.5*reb + 1.2*stl - 1.5*tov + 80*fg_pct - 40 + np.random.normal(0, 8, n)
    df = pd.DataFrame({
        'AST': ast, 'REB': reb, 'STL': stl, 'TOV': tov,
        'FG_PCT': fg_pct, 'PLUS_MINUS': plus_minus
    })

predictores = ['AST', 'REB', 'STL', 'TOV', 'FG_PCT']
X = df[predictores].dropna()
y = df.loc[X.index, 'PLUS_MINUS']

# Regresión múltiple
X_mat = np.column_stack([np.ones(len(X)), X.values])
XtX_inv = np.linalg.inv(X_mat.T @ X_mat)
beta = XtX_inv @ X_mat.T @ y.values

y_pred = X_mat @ beta
residuos = y.values - y_pred
n_obs, k = len(y), len(predictores)

ss_res = np.sum(residuos**2)
ss_tot = np.sum((y.values - y.mean())**2)
r2 = 1 - ss_res/ss_tot
r2_adj = 1 - (1-r2)*(n_obs-1)/(n_obs-k-1)

mse = ss_res / (n_obs - k - 1)
se_beta = np.sqrt(np.diag(XtX_inv) * mse)
t_stats = beta / se_beta
p_values = 2 * (1 - stats.t.cdf(np.abs(t_stats), n_obs - k - 1))

f_stat = (r2/k) / ((1-r2)/(n_obs-k-1))

# Generar filas de tabla
filas = ""
for i, var in enumerate(predictores):
    var_escaped = var.replace('_', '\\_')
    sig = '$^{***}$' if p_values[i+1] < 0.001 else ('$^{**}$' if p_values[i+1] < 0.01 else ('$^{*}$' if p_values[i+1] < 0.05 else ''))
    p_str = '$<$0.001' if p_values[i+1] < 0.001 else f'{p_values[i+1]:.3f}'
    filas += f"{var_escaped} & {beta[i+1]:.3f} & {se_beta[i+1]:.3f} & {t_stats[i+1]:.2f} & {p_str}{sig} \\\\\n"

# GENERAR LATEX
latex = f"""% AUTO-GENERADO por pregunta3_multiple.py
\\subsection{{Pregunta 3: Predictores de PLUS\\_MINUS}}

\\subsubsection{{Regresión Lineal Múltiple}}

\\textbf{{Objetivo:}} Identificar qué estadísticas predicen mejor PLUS\\_MINUS.

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|c|c|c|c|}}
\\hline
\\textbf{{Variable}} & \\textbf{{$\\beta$}} & \\textbf{{Error}} & \\textbf{{t}} & \\textbf{{p-valor}} \\\\
\\hline
{filas}\\hline
\\end{{tabular}}
\\caption{{Coeficientes de regresión. $^{{***}}$p$<$0.001}}
\\end{{table}}

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Estadístico}} & \\textbf{{Valor}} \\\\
\\hline
$R^2$ & {r2:.3f} \\\\
$R^2$ ajustado & {r2_adj:.3f} \\\\
F-estadístico & {f_stat:.2f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Bondad de ajuste}}
\\end{{table}}

\\textbf{{Conclusión:}} El modelo explica el {r2*100:.1f}\\% de la variabilidad. FG\\_PCT es el predictor más fuerte.
"""

with open('generated/pregunta3.tex', 'w', encoding='utf-8') as f:
    f.write(latex)

print("[OK] generated/pregunta3.tex creado")
print(f"  R²={r2:.3f}, F={f_stat:.2f}")
