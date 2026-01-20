# ===============================
# 📌 Preprocesamiento Completo NBA 2010-2024
# ===============================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler

# -------------------------------------------------
# 1️⃣ Cargar datos
# -------------------------------------------------
# Descomenta si aún no has cargado el dataframe
df = pd.read_csv("nba_2010_2024.csv")  

# Crear copia de trabajo para no alterar el original
df_prep = df.copy()

# -------------------------------------------------
# 2️⃣ Codificación binaria de la variable objetivo
# -------------------------------------------------
# 'W' -> 1, 'L' -> 0 para modelos de clasificación
df_prep["WL_bin"] = df_prep["WL"].map({"W": 1, "L": 0})

# -------------------------------------------------
# 3️⃣ Variables numéricas para Z-score y Min-Max
# -------------------------------------------------
vars_zscore = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV']   # Variables a estandarizar
vars_minmax = ['FG_PCT', 'FG3_PCT', 'FT_PCT']             # Variables a normalizar

# -------------------------------
# 3a️⃣ Estandarización Z-score
# -------------------------------
scaler_z = StandardScaler()
df_prep[vars_zscore] = scaler_z.fit_transform(df_prep[vars_zscore])

# -------------------------------
# 3b️⃣ Normalización Min-Max
# -------------------------------
scaler_mm = MinMaxScaler()
df_prep[vars_minmax] = scaler_mm.fit_transform(df_prep[vars_minmax])

# -------------------------------------------------
# 4️⃣ Tratamiento de outliers (Winsorizing)
# -------------------------------------------------
# Solo para PLUS_MINUS, los otros se mantienen
q05 = df_prep['PLUS_MINUS'].quantile(0.05)
q95 = df_prep['PLUS_MINUS'].quantile(0.95)
df_prep['PLUS_MINUS_W'] = df_prep['PLUS_MINUS'].clip(lower=q05, upper=q95)

# -------------------------------------------------
# 5️⃣ Creación de Variables Derivadas Avanzadas
# -------------------------------------------------
# Capturan métricas de rendimiento avanzado
df_prep['POSS'] = df_prep['FGA'] - df_prep['OREB'] + df_prep['TOV'] + 0.4 * df_prep['FTA']
df_prep['PACE'] = (df_prep['POSS'] / df_prep['MIN']) * 48
df_prep['OFF_EFF'] = (df_prep['PTS'] / df_prep['POSS']) * 100
df_prep['AST_TO_RATIO'] = df_prep['AST'] / df_prep['TOV']
df_prep['FG3A_RATE'] = df_prep['FG3A'] / df_prep['FGA']
df_prep['EFFICIENT_SCORING'] = df_prep['FG_PCT'] + 0.5 * df_prep['FG3_PCT']

# -------------------------------------------------
# 6️⃣ Vista rápida de variables procesadas
# -------------------------------------------------
print("✅ Primeros 5 registros de variables Z-score:\n", df_prep[vars_zscore].head())
print("\n✅ Primeros 5 registros de variables Min-Max:\n", df_prep[vars_minmax].head())
print("\n✅ Variable objetivo codificada (WL_bin):\n", df_prep[['WL', 'WL_bin']].head())
print("\n✅ PLUS_MINUS Winsorizado:\n", df_prep[['PLUS_MINUS', 'PLUS_MINUS_W']].head())
print("\n✅ Primeros 5 registros de variables derivadas:\n", 
      df_prep[['POSS','PACE','OFF_EFF','AST_TO_RATIO','FG3A_RATE','EFFICIENT_SCORING']].head())

# -------------------------------------------------
# 7️⃣ Estadísticos descriptivos rápidos
# -------------------------------------------------
estad_desc = df_prep[['PTS','REB','AST','FG_PCT','PLUS_MINUS']].describe().T
estad_desc['Coef_Var'] = estad_desc['std'] / estad_desc['mean'] * 100  # Coeficiente de variación %
print("\n✅ Estadísticos descriptivos principales:\n", estad_desc)

# -------------------------------------------------
# 8️⃣ Opcional: guardar dataframe final preprocesado
# -------------------------------------------------
df_prep.to_csv("nba_2010_2024_preprocesado_completo.csv", index=False)
