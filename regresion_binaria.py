import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt

# =========================
# 1. Cargar datos
# =========================
df = pd.read_csv("nba_2010_2024_preprocesado_completo.csv")

# =========================
# 2. Preparar variables
# =========================
data = df[['FG3M', 'FG_PCT', 'WL']].copy()
data['WL_BINARY'] = data['WL'].map({'W': 1, 'L': 0})
data = data.dropna()

X = data[['FG3M', 'FG_PCT']]
y = data['WL_BINARY']

# Agregar intercepto
X_const = sm.add_constant(X)

# =========================
# 3. Regresión logística binaria
# =========================
model = sm.Logit(y, X_const)
result = model.fit()

print(result.summary())

# Odds Ratios
print("\nOdds Ratios:")
print(np.exp(result.params))

# =========================
# 4. Gráfico 1: Probabilidad vs FG3M
# =========================
fg3m_range = np.linspace(X['FG3M'].min(), X['FG3M'].max(), 100)
fg_pct_mean = X['FG_PCT'].mean()

X_pred = pd.DataFrame({
    'const': 1,
    'FG3M': fg3m_range,
    'FG_PCT': fg_pct_mean
})

prob_win = result.predict(X_pred)

plt.figure()
plt.plot(fg3m_range, prob_win)
plt.xlabel("Triples anotados (FG3M)")
plt.ylabel("Probabilidad de victoria")
plt.title("Probabilidad de victoria vs Triples anotados\n(controlando FG_PCT)")
plt.show()

# =========================
# 5. Gráfico 2: Distribución de probabilidades
# =========================
data['prob_predicha'] = result.predict(X_const)

plt.figure()
plt.hist(data.loc[data['WL_BINARY'] == 0, 'prob_predicha'],
         bins=20, alpha=0.5, label='Derrotas')
plt.hist(data.loc[data['WL_BINARY'] == 1, 'prob_predicha'],
         bins=20, alpha=0.5, label='Victorias')
plt.xlabel("Probabilidad predicha de victoria")
plt.ylabel("Frecuencia")
plt.title("Distribución de probabilidades predichas")
plt.legend()
plt.show()

# =========================
# 6. Conclusión automática
# =========================
p_fg3m = result.pvalues['FG3M']

print("\nConclusión:")
if p_fg3m < 0.05:
    print("Existe una relación estadísticamente significativa entre FG3M y la victoria,")
    print("controlando por la efectividad general de tiro (FG_PCT).")
else:
    print("No se encuentra una relación estadísticamente significativa entre FG3M y la victoria.")
