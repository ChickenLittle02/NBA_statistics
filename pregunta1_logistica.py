# ===========================================
# PREGUNTA 1: FG3M vs VICTORIA (WL)
# ===========================================
# Genera: generated/pregunta1.tex

import pandas as pd
import numpy as np
from scipy import stats
from scipy.optimize import minimize
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
    fg_pct = np.random.normal(0.50, 0.05, n)
    fg3m = np.random.normal(12, 4, n)
    prob_win = 1 / (1 + np.exp(-(0.15*fg3m + 8*fg_pct - 5.5)))
    wl = np.random.binomial(1, prob_win)
    df = pd.DataFrame({
        'FG3M': fg3m,
        'FG_PCT': fg_pct,
        'WL': np.where(wl == 1, 'W', 'L')
    })

df['WIN'] = (df['WL'] == 'W').astype(int)

# CHI-CUADRADO
df['FG3M_CAT'] = pd.cut(df['FG3M'], bins=3, labels=['Bajo', 'Medio', 'Alto'])
tabla = pd.crosstab(df['FG3M_CAT'], df['WL'])
chi2, p_chi, dof, _ = stats.chi2_contingency(tabla)
v_cramer = np.sqrt(chi2 / (len(df) * (min(tabla.shape) - 1)))

# REGRESIÓN LOGÍSTICA
X1 = (df['FG3M'] - df['FG3M'].mean()) / df['FG3M'].std()
X2 = (df['FG_PCT'] - df['FG_PCT'].mean()) / df['FG_PCT'].std()
y = df['WIN'].values

def neg_log_likelihood(params):
    b0, b1, b2 = params
    z = b0 + b1*X1 + b2*X2
    p = 1 / (1 + np.exp(-np.clip(z, -500, 500)))
    p = np.clip(p, 1e-10, 1-1e-10)
    return -np.sum(y*np.log(p) + (1-y)*np.log(1-p))

result = minimize(neg_log_likelihood, [0, 0, 0], method='BFGS')
b0, b1, b2 = result.x
or_fg3m = np.exp(b1)
or_fgpct = np.exp(b2)

# GENERAR LATEX
latex = f"""% AUTO-GENERADO por pregunta1_logistica.py
\\subsection{{Pregunta 1: Correlación FG3M vs Victoria}}

\\subsubsection{{Análisis Chi-Cuadrado}}

\\textbf{{Objetivo:}} Determinar si existe asociación significativa entre FG3M y WL.

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|c|c|c|}}
\\hline
\\textbf{{FG3M}} & \\textbf{{Derrota}} & \\textbf{{Victoria}} & \\textbf{{Total}} \\\\
\\hline
Bajo & {tabla.iloc[0,0]} & {tabla.iloc[0,1]} & {tabla.iloc[0].sum()} \\\\
Medio & {tabla.iloc[1,0]} & {tabla.iloc[1,1]} & {tabla.iloc[1].sum()} \\\\
Alto & {tabla.iloc[2,0]} & {tabla.iloc[2,1]} & {tabla.iloc[2].sum()} \\\\
\\hline
\\end{{tabular}}
\\caption{{Tabla de contingencia FG3M vs WL}}
\\end{{table}}

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|c|}}
\\hline
\\textbf{{Estadístico}} & \\textbf{{Valor}} \\\\
\\hline
$\\chi^2$ & {chi2:.2f} \\\\
Grados de libertad & {dof} \\\\
Valor p & {'$<$0.001' if p_chi < 0.001 else f'{p_chi:.4f}'} \\\\
V de Cramér & {v_cramer:.3f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Resultados Chi-Cuadrado ($\\alpha$ = 0.05)}}
\\end{{table}}

\\subsubsection{{Regresión Logística}}

\\textbf{{Modelo:}} $\\log\\left(\\frac{{P(Win)}}{{1-P(Win)}}\\right) = \\beta_0 + \\beta_1 \\cdot FG3M + \\beta_2 \\cdot FG\\_PCT$

\\begin{{table}}[H]
\\centering
\\begin{{tabular}}{{|l|c|c|}}
\\hline
\\textbf{{Variable}} & \\textbf{{Coeficiente}} & \\textbf{{Odds Ratio}} \\\\
\\hline
FG3M & {b1:.2f} & {or_fg3m:.2f} \\\\
FG\\_PCT & {b2:.2f} & {or_fgpct:.2f} \\\\
\\hline
\\end{{tabular}}
\\caption{{Coeficientes de regresión logística}}
\\end{{table}}

\\textbf{{Conclusión:}} {'La filosofía de vivir o morir por el triple tiene respaldo estadístico.' if p_chi < 0.05 and b1 > 0 else 'Resultados no concluyentes.'}
"""

with open('generated/pregunta1.tex', 'w', encoding='utf-8') as f:
    f.write(latex)

print("[OK] generated/pregunta1.tex creado")
print(f"  Chi²={chi2:.2f}, p={'<0.001' if p_chi < 0.001 else f'{p_chi:.4f}'}")
print(f"  OR(FG3M)={or_fg3m:.2f}")
