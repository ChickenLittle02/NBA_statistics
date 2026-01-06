# ==========================================
# ANALISIS EXPLORATORIO DE DATOS (AED) - NBA
# ==========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os

# 1. CONFIGURACIÓN Y CARGA DE DATOS
# ----------------------------------------------------------------
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

FILE_NAME = 'regular_season_totals_2010_2024.csv'

# Simulación de datos si el archivo no existe (Para fines de prueba)
if not os.path.exists(FILE_NAME):
    print(f"(!) Archivo {FILE_NAME} no encontrado. Creando datos de prueba...")
    data_demo = {
        'TEAM_ABBREVIATION': np.random.choice(['LAL', 'GSW', 'BOS', 'CHI', 'MIA'], 1000),
        'WL': np.random.choice(['W', 'L'], 1000),
        'MIN': np.random.normal(240, 5, 1000),
        'PTS': np.random.normal(110, 15, 1000),
        'REB': np.random.normal(45, 10, 1000),
        'AST': np.random.normal(25, 5, 1000),
        'FG_PCT': np.random.uniform(0.4, 0.55, 1000),
        'FG3_PCT': np.random.uniform(0.3, 0.45, 1000),
        'FT_PCT': np.random.uniform(0.7, 0.9, 1000),
        'PLUS_MINUS': np.random.normal(0, 10, 1000),
        'MATCHUP': ['Matchup Demo'] * 1000
    }
    df = pd.DataFrame(data_demo)
else:
    df = pd.read_csv(FILE_NAME)

print(f"Dataset cargado. Dimensiones: {df.shape}")

# 2. CLASIFICACIÓN DE VARIABLES
# ----------------------------------------------------------------
print("\n" + "="*30 + " CLASIFICACIÓN DE VARIABLES " + "="*30)
variables_cualitativas = []
variables_cuantitativas_discretas = []
variables_cuantitativas_continuas = []

for col in df.columns:
    dtype = df[col].dtype
    if dtype == 'object' or df[col].nunique() < 10:
        variables_cualitativas.append(col)
    else:
        unique_vals = df[col].dropna().nunique()
        if pd.api.types.is_integer_dtype(dtype) and unique_vals < 50:
            variables_cuantitativas_discretas.append(col)
        else:
            variables_cuantitativas_continuas.append(col)

print(f"Cualitativas: {variables_cualitativas}")
print(f"Continuas: {variables_cuantitativas_continuas[:5]}...")

# 3. ANÁLISIS DESCRIPTIVO (CUANTITATIVAS)
# ----------------------------------------------------------------
variables_analisis = [v for v in ['MIN', 'PTS', 'REB', 'AST', 'FG_PCT', 'PLUS_MINUS'] if v in df.columns]
estadisticas_df = pd.DataFrame(index=['Media', 'Mediana', 'Moda', 'Desv. Estándar', 'Varianza', 'Min', 'Max', 'IQR'])

for var in variables_analisis:
    datos = df[var].dropna()
    media = np.mean(datos)
    mediana = np.median(datos)
    moda = stats.mode(datos, keepdims=True).mode[0]
    desv_std = np.std(datos, ddof=1)
    q1, q3 = np.percentile(datos, [25, 75])
    
    estadisticas_df[var] = [
        f"{media:.2f}", f"{mediana:.2f}", f"{moda:.2f}",
        f"{desv_std:.2f}", f"{np.var(datos, ddof=1):.2f}",
        f"{np.min(datos):.2f}", f"{np.max(datos):.2f}", f"{q3-q1:.2f}"
    ]

print("\nESTADÍSTICAS DESCRIPTIVAS:")
display(estadisticas_df)

# 4. TABLAS DE FRECUENCIA (TDEF)
# ----------------------------------------------------------------
def crear_tdef(datos, n_intervals=8):
    datos_clean = datos.dropna()
    counts, bins = np.histogram(datos_clean, bins=n_intervals)
    intervals = [f"[{bins[i]:.2f}, {bins[i+1]:.2f})" for i in range(len(bins)-1)]
    
    tdef = pd.DataFrame({
        'Intervalo': intervals,
        'Frec. Abs (ni)': counts,
        'Frec. Abs Acum (Ni)': np.cumsum(counts),
        'Frec. Rel %': (counts / len(datos_clean) * 100).round(2)
    })
    return tdef

if 'PTS' in df.columns:
    print("\nTDEF - PUNTOS (PTS):")
    display(crear_tdef(df['PTS']))

# 5. VISUALIZACIÓN
# ----------------------------------------------------------------
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Histograma PTS
sns.histplot(df['PTS'], kde=True, ax=axes[0,0], color='skyblue')
axes[0,0].set_title('Distribución de Puntos')

# Boxplot PTS por WL
if 'WL' in df.columns:
    sns.boxplot(x='WL', y='PTS', data=df, ax=axes[0,1])
    axes[0,1].set_title('Puntos según Resultado (W/L)')

# Top Equipos
df['TEAM_ABBREVIATION'].value_counts().head(10).plot(kind='bar', ax=axes[1,0], color='salmon')
axes[1,0].set_title('Top 10 Equipos en Dataset')

# Correlación
sns.heatmap(df[variables_analisis].corr(), annot=True, cmap='coolwarm', ax=axes[1,1])
axes[1,1].set_title('Mapa de Correlación')

plt.tight_layout()
plt.show()

# 6. ANÁLISIS DE OUTLIERS (IQR)
# ----------------------------------------------------------------
print("\n" + "="*30 + " ANÁLISIS DE OUTLIERS " + "="*30)
for var in ['PTS', 'REB']:
    if var in df.columns:
        q1, q3 = np.percentile(df[var].dropna(), [25, 75])
        iqr = q3 - q1
        outliers = df[(df[var] < (q1 - 1.5*iqr)) | (df[var] > (q3 + 1.5*iqr))]
        print(f"Variable {var}: {len(outliers)} outliers detectados ({len(outliers)/len(df)*100:.2f}%)")

# 7. EXPORTACIÓN
# ----------------------------------------------------------------
estadisticas_df.to_csv('resumen_estadistico_nba.csv')
print("\n>>> Proceso finalizado. Archivo 'resumen_estadistico_nba.csv' guardado.")