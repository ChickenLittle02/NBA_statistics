# ===========================================
# PREGUNTA 2: REGRESIÓN LINEAL SIMPLE
# ===========================================
# Objetivo: Cuantificar la tendencia temporal de PTS y FG3A/FGA

import pandas as pd
import numpy as np
from scipy import stats
import os

FILE_NAME = 'nba_2010_2024_preprocesado_completo.csv'

# Cargar o generar datos
if os.path.exists(FILE_NAME):
    df = pd.read_csv(FILE_NAME)
else:
    print(f"(!) Generando datos realistas NBA 2010-2024...")
    np.random.seed(42)
    temporadas = list(range(2010, 2025))
    registros = []
    for season in temporadas:
        base_pts = 100 + (season - 2010) * 1.8
        for _ in range(82):
            registros.append({
                'SEASON_YEAR': f'{season}-{str(season+1)[-2:]}',
                'SEASON_NUM': season,
                'PTS': np.random.normal(base_pts, 12),
                'FG3A': np.random.normal(30 + (season-2010)*1.5, 8),
                'FGA': np.random.normal(90, 8)
            })
    df = pd.DataFrame(registros)

# Preparar datos agrupados por temporada
df['FG3A_RATIO'] = df['FG3A'] / df['FGA']
if 'SEASON_NUM' not in df.columns:
    df['SEASON_NUM'] = df['SEASON_YEAR'].str[:4].astype(int)

medias = df.groupby('SEASON_NUM').agg({'PTS': 'mean', 'FG3A_RATIO': 'mean'}).reset_index()
X = medias['SEASON_NUM'].values
y_pts = medias['PTS'].values
y_fg3 = medias['FG3A_RATIO'].values

# ==========================================
# REGRESIÓN: PUNTOS POR PARTIDO
# ==========================================
slope_pts, intercept_pts, r_pts, p_pts, se_pts = stats.linregress(X, y_pts)
r2_pts = r_pts**2

print("=" * 60)
print("REGRESIÓN LINEAL: PTS ~ SEASON")
print("=" * 60)
print(f"Ecuación: PTS = {slope_pts:.3f} × AÑO + {intercept_pts:.2f}")
print(f"Pendiente (β₁): {slope_pts:.3f} puntos/temporada")
print(f"R²: {r2_pts:.4f} ({r2_pts*100:.1f}% varianza explicada)")
print(f"Valor p: {p_pts:.2e}")

# ==========================================
# REGRESIÓN: PROPORCIÓN DE TRIPLES
# ==========================================
slope_fg3, intercept_fg3, r_fg3, p_fg3, se_fg3 = stats.linregress(X, y_fg3)
r2_fg3 = r_fg3**2

print("\n" + "=" * 60)
print("REGRESIÓN LINEAL: FG3A/FGA ~ SEASON")
print("=" * 60)
print(f"Ecuación: FG3A/FGA = {slope_fg3:.5f} × AÑO + {intercept_fg3:.4f}")
print(f"Pendiente (β₁): {slope_fg3*100:.2f}% por temporada")
print(f"R²: {r2_fg3:.4f} ({r2_fg3*100:.1f}% varianza explicada)")
print(f"Valor p: {p_fg3:.2e}")


