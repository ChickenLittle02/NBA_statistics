# =============================================
# 1. CONFIGURACIÓN INICIAL E IMPORTACIÓN DE LIBRERÍAS
# =============================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
from IPython.display import display, HTML
from statsmodels.stats.outliers_influence import variance_inflation_factor
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from scipy.stats.mstats import winsorize
from scipy.stats import levene

# Configuración de estilo y advertencias
warnings.filterwarnings('ignore')
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

# Configuración para mostrar todas las columnas en pandas
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)

print("Librerías importadas correctamente")

# =============================================
# 2. CARGA DE DATOS
# =============================================

# Cargar el dataset
df = pd.read_csv('regular_season_totals_2010_2024.csv', low_memory=False)

# Mostrar información básica del dataset
print("=" * 80)
print("INFORMACIÓN BÁSICA DEL DATASET")
print("=" * 80)
print(f"Dimensiones del dataset: {df.shape}")
print(f"Número de filas: {df.shape[0]}")
print(f"Número de columnas: {df.shape[1]}")

# Mostrar primeras filas
print("\nPrimeras 5 filas del dataset:")
display(df.head())

# =============================================
# 3. LIMPIEZA Y PREPROCESAMIENTO DE DATOS
# =============================================

# Identificar valores nulos
print("=" * 80)
print("ANÁLISIS DE VALORES NULOS")
print("=" * 80)
null_counts = df.isnull().sum()
null_percentage = (null_counts / len(df)) * 100
null_summary = pd.DataFrame({
    'Valores Nulos': null_counts,
    'Porcentaje': null_percentage
})
null_summary = null_summary[null_summary['Valores Nulos'] > 0].sort_values('Porcentaje', ascending=False)

if len(null_summary) > 0:
    display(null_summary)
else:
    print("¡No hay valores nulos en el dataset!")

# Identificar columnas con muchos valores únicos (posibles IDs)
print("\nColumnas con alto número de valores únicos:")
for col in df.columns:
    unique_count = df[col].nunique()
    if unique_count > 100:
        print(f"{col}: {unique_count} valores únicos ({unique_count/len(df)*100:.1f}%)")

# Convertir columnas de fecha
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df['SEASON_YEAR'] = df['SEASON_YEAR'].astype(str)

# Extraer año de la temporada como número
df['SEASON_YEAR_NUM'] = df['SEASON_YEAR'].str.extract(r'(\d{4})').astype(float)

# =============================================
# 4. ANÁLISIS EXPLORATORIO - VARIABLES CLAVE
# =============================================

# Seleccionar variables numéricas clave para análisis
key_metrics = [
    'PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV', 'FGM', 'FGA', 
    'FG3M', 'FG3A', 'FTM', 'FTA', 'PLUS_MINUS', 'MIN',
    'FG_PCT', 'FG3_PCT', 'FT_PCT'
]

# Filtrar solo columnas que existen en el dataset
existing_metrics = [col for col in key_metrics if col in df.columns]
print(f"\nVariables clave encontradas: {len(existing_metrics)} de {len(key_metrics)}")
print(f"Variables: {existing_metrics}")

# =============================================
# 5. CÁLCULO DE MEDIDAS DE TENDENCIA CENTRAL Y DISPERSIÓN
# =============================================

print("\n" + "=" * 80)
print("MEDIDAS DE TENDENCIA CENTRAL Y DISPERSIÓN")
print("=" * 80)

# Función para calcular todas las medidas estadísticas
def calcular_medidas_estadisticas(data, variable):
    """Calcula medidas de tendencia central y dispersión para una variable"""
    
    # Filtrar valores no nulos
    valores = data[variable].dropna()
    
    if len(valores) == 0:
        return None
    
    # Medidas de tendencia central
    medidas = {
        'Variable': variable,
        'N': len(valores),
        'Mínimo': valores.min(),
        'Máximo': valores.max(),
        'Rango': valores.max() - valores.min(),
        'Media': valores.mean(),
        'Mediana': valores.median(),
        'Moda': valores.mode().iloc[0] if not valores.mode().empty else np.nan,
        'Q1 (25%)': valores.quantile(0.25),
        'Q2 (50%)': valores.quantile(0.50),
        'Q3 (75%)': valores.quantile(0.75),
        'IQR (Q3-Q1)': valores.quantile(0.75) - valores.quantile(0.25),
        'Varianza': valores.var(),
        'Desviación Estándar': valores.std(),
        'Coef. Variación': (valores.std() / valores.mean()) * 100 if valores.mean() != 0 else np.nan,
        'Asimetría (Skewness)': valores.skew(),
        'Curtosis': valores.kurtosis()
    }
    
    return medidas

# Calcular medidas para todas las variables clave
resultados_estadisticos = []

for variable in existing_metrics:
    medidas = calcular_medidas_estadisticas(df, variable)
    if medidas:
        resultados_estadisticos.append(medidas)

# Crear DataFrame con resultados
df_estadisticas = pd.DataFrame(resultados_estadisticos)

# Formatear valores para mejor visualización
columnas_numericas = df_estadisticas.columns.drop(['Variable'])
df_estadisticas[columnas_numericas] = df_estadisticas[columnas_numericas].applymap(
    lambda x: f"{x:,.3f}" if isinstance(x, (int, float)) else x
)

print("\nMedidas estadísticas completas para todas las variables:")
display(df_estadisticas)

# =============================================
# 6. ANÁLISIS COMPARATIVO DE MEDIDAS ESTADÍSTICAS
# =============================================

print("\n" + "=" * 80)
print("ANÁLISIS COMPARATIVO DE MEDIDAS ESTADÍSTICAS")
print("=" * 80)

# Crear resumen comparativo de las 5 variables principales
variables_principales = ['PTS', 'AST', 'REB', 'FG_PCT', 'PLUS_MINUS']
variables_principales = [v for v in variables_principales if v in existing_metrics]

if variables_principales:
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    
    for i, variable in enumerate(variables_principales[:6]):
        ax = axes[i]
        
        # Obtener valores
        valores = df[variable].dropna()
        
        # Crear gráfico de caja con puntos
        bp = ax.boxplot(valores, vert=True, patch_artist=True, showmeans=True)
        
        # Personalizar
        bp['boxes'][0].set_facecolor('lightblue')
        bp['medians'][0].set_color('red')
        bp['medians'][0].set_linewidth(2)
        bp['means'][0].set_color('green')
        bp['means'][0].set_linewidth(2)
        
        # Calcular medidas
        media = valores.mean()
        mediana = valores.median()
        desv_std = valores.std()
        
        # Añadir texto con medidas
        texto = f"Media: {media:.2f}\nMediana: {mediana:.2f}\nDesv. Est.: {desv_std:.2f}"
        ax.text(0.05, 0.95, texto, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        ax.set_title(f'Distribución de {variable}', fontsize=14)
        ax.set_ylabel(variable)
        ax.grid(True, alpha=0.3)
    
    # Ajustar layout
    plt.tight_layout()
    plt.show()

# =============================================
# 7. TABLAS DE DISTRIBUCIÓN EMPÍRICA DE FRECUENCIA
# =============================================

print("\n" + "=" * 80)
print("TABLAS DE DISTRIBUCIÓN EMPÍRICA DE FRECUENCIA")
print("=" * 80)

def crear_tabla_frecuencias(data, variable, n_bins=10):
    """Crea tabla de distribución de frecuencias para una variable"""
    
    # Filtrar valores no nulos
    valores = data[variable].dropna()
    
    if len(valores) == 0:
        return None
    
    # Calcular número óptimo de bins (regla de Sturges)
    n = len(valores)
    sturges_bins = int(1 + 3.322 * np.log10(n))
    n_bins = min(n_bins, sturges_bins)
    
    # Crear intervalos
    min_val = valores.min()
    max_val = valores.max()
    ancho_intervalo = (max_val - min_val) / n_bins
    
    # Definir límites de los intervalos
    limites = np.linspace(min_val, max_val, n_bins + 1)
    
    # Crear etiquetas de intervalos
    etiquetas = []
    for i in range(len(limites) - 1):
        etiquetas.append(f"[{limites[i]:.1f}, {limites[i+1]:.1f})")
    
    # Crear tabla de frecuencias
    frec_abs = pd.cut(valores, bins=limites, right=False, labels=etiquetas).value_counts().sort_index()
    frec_rel = (frec_abs / n * 100).round(2)
    frec_abs_acum = frec_abs.cumsum()
    frec_rel_acum = frec_rel.cumsum()
    
    # Crear DataFrame
    tabla = pd.DataFrame({
        'Intervalo': etiquetas,
        'Frecuencia Absoluta': frec_abs.values,
        'Frecuencia Relativa (%)': frec_rel.values,
        'Frecuencia Absoluta Acumulada': frec_abs_acum.values,
        'Frecuencia Relativa Acumulada (%)': frec_rel_acum.values
    })
    
    # Añadir información resumen
    resumen = {
        'Variable': variable,
        'Número de observaciones': n,
        'Número de intervalos': n_bins,
        'Ancho del intervalo': ancho_intervalo,
        'Mínimo': min_val,
        'Máximo': max_val,
        'Amplitud total': max_val - min_val
    }
    
    return tabla, resumen

# Crear tablas de frecuencia para variables seleccionadas
variables_tabla = ['PTS', 'AST', 'REB', 'FG_PCT', 'PLUS_MINUS']
variables_tabla = [v for v in variables_tabla if v in existing_metrics]

for variable in variables_tabla:
    print(f"\n{'='*60}")
    print(f"TABLA DE DISTRIBUCIÓN DE FRECUENCIAS: {variable}")
    print(f"{'='*60}")
    
    resultado = crear_tabla_frecuencias(df, variable, n_bins=8)
    
    if resultado:
        tabla, resumen = resultado
        
        # Mostrar resumen
        print("\nRESUMEN:")
        for clave, valor in resumen.items():
            if isinstance(valor, float):
                print(f"  {clave}: {valor:.3f}")
            else:
                print(f"  {clave}: {valor}")
        
        # Mostrar tabla
        print(f"\nTABLA DE FRECUENCIAS (primeras 10 filas):")
        display(tabla.head(10))
        
        # Visualizar histograma con tabla de frecuencias
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        
        # Histograma
        valores = df[variable].dropna()
        axes[0].hist(valores, bins=8, edgecolor='black', alpha=0.7, color='skyblue')
        axes[0].axvline(valores.mean(), color='red', linestyle='--', linewidth=2, label=f'Media: {valores.mean():.2f}')
        axes[0].axvline(valores.median(), color='green', linestyle='--', linewidth=2, label=f'Mediana: {valores.median():.2f}')
        axes[0].set_title(f'Histograma de {variable}', fontsize=14)
        axes[0].set_xlabel(variable)
        axes[0].set_ylabel('Frecuencia')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Polígono de frecuencias acumuladas
        ejes_intervalos = np.arange(len(tabla))
        axes[1].plot(ejes_intervalos, tabla['Frecuencia Relativa Acumulada (%)'], 
                    marker='o', linewidth=2, markersize=8, color='orange')
        axes[1].fill_between(ejes_intervalos, 0, tabla['Frecuencia Relativa Acumulada (%)'], 
                            alpha=0.3, color='orange')
        axes[1].set_title(f'Polígono de Frecuencias Acumuladas: {variable}', fontsize=14)
        axes[1].set_xlabel('Intervalos')
        axes[1].set_ylabel('Frecuencia Relativa Acumulada (%)')
        axes[1].set_xticks(ejes_intervalos)
        axes[1].set_xticklabels([f'I{i+1}' for i in ejes_intervalos], rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
        
        # Crear tabla de resumen estadístico detallado
        print(f"\nRESUMEN ESTADÍSTICO DETALLADO: {variable}")
        print("-" * 50)
        
        # Calcular percentiles
        percentiles = [5, 10, 25, 50, 75, 90, 95]
        percentiles_valores = np.percentile(valores, percentiles)
        
        resumen_detallado = pd.DataFrame({
            'Percentil': [f'P{p}' for p in percentiles],
            'Valor': [f"{v:.3f}" for v in percentiles_valores],
            'Interpretación': [
                f"{percentiles[0]}% de los valores son ≤ {percentiles_valores[0]:.3f}",
                f"{percentiles[1]}% de los valores son ≤ {percentiles_valores[1]:.3f}",
                "Primer cuartil (Q1)",
                "Mediana (Q2)",
                "Tercer cuartil (Q3)",
                f"{percentiles[5]}% de los valores son ≤ {percentiles_valores[5]:.3f}",
                f"{percentiles[6]}% de los valores son ≤ {percentiles_valores[6]:.3f}"
            ]
        })
        
        display(resumen_detallado)

# =============================================
# 8. ANÁLISIS DE DISTRIBUCIÓN POR PERCENTILES
# =============================================

print("\n" + "=" * 80)
print("ANÁLISIS DE DISTRIBUCIÓN POR PERCENTILES")
print("=" * 80)

# Función para crear análisis de percentiles para múltiples variables
def analisis_percentiles_variables(data, variables, percentiles=[10, 25, 50, 75, 90]):
    """Crea análisis de percentiles para múltiples variables"""
    
    resultados = {}
    
    for variable in variables:
        if variable in data.columns:
            valores = data[variable].dropna()
            
            if len(valores) > 0:
                percentiles_calculados = {}
                
                for p in percentiles:
                    percentiles_calculados[f'P{p}'] = np.percentile(valores, p)
                
                # Añadir medidas adicionales
                percentiles_calculados['Media'] = valores.mean()
                percentiles_calculados['Desv. Est.'] = valores.std()
                percentiles_calculados['CV (%)'] = (valores.std() / valores.mean() * 100) if valores.mean() != 0 else np.nan
                
                resultados[variable] = percentiles_calculados
    
    # Crear DataFrame
    df_percentiles = pd.DataFrame(resultados).T
    
    # Formatear valores
    for col in df_percentiles.columns:
        df_percentiles[col] = df_percentiles[col].apply(lambda x: f"{x:.3f}" if isinstance(x, (int, float)) else x)
    
    return df_percentiles

# Aplicar análisis de percentiles a variables clave
if existing_metrics:
    df_percentiles = analisis_percentiles_variables(df, existing_metrics[:8])
    print("\nAnálisis de percentiles para variables clave:")
    display(df_percentiles)

# =============================================
# 9. VISUALIZACIONES - HISTOGRAMAS CON MEDIDAS ESTADÍSTICAS
# =============================================

print("\n" + "=" * 80)
print("HISTOGRAMAS CON MEDIDAS ESTADÍSTICAS")
print("=" * 80)

# Crear histogramas detallados con medidas estadísticas
variables_para_histograma = existing_metrics[:8]

fig, axes = plt.subplots(2, 4, figsize=(20, 10))
axes = axes.flatten()

for i, variable in enumerate(variables_para_histograma):
    if i < len(axes):
        ax = axes[i]
        valores = df[variable].dropna()
        
        # Crear histograma
        n, bins, patches = ax.hist(valores, bins=20, alpha=0.7, edgecolor='black', color='skyblue', density=False)
        
        # Calcular medidas
        media = valores.mean()
        mediana = valores.median()
        moda = valores.mode().iloc[0] if not valores.mode().empty else np.nan
        desv_std = valores.std()
        
        # Añadir líneas de referencia
        ax.axvline(media, color='red', linestyle='-', linewidth=2, label=f'Media: {media:.2f}')
        ax.axvline(mediana, color='green', linestyle='--', linewidth=2, label=f'Mediana: {mediana:.2f}')
        ax.axvline(moda, color='orange', linestyle='-.', linewidth=2, label=f'Moda: {moda:.2f}')
        
        # Añadir área de ±1 desviación estándar
        ax.axvspan(media - desv_std, media + desv_std, alpha=0.2, color='gray', label=f'±1 σ: {desv_std:.2f}')
        
        # Añadir curva de densidad KDE
        from scipy.stats import gaussian_kde
        if len(valores) > 1:
            kde = gaussian_kde(valores)
            x_vals = np.linspace(valores.min(), valores.max(), 100)
            ax.plot(x_vals, kde(x_vals) * len(valores) * (bins[1] - bins[0]), 
                   color='darkblue', linewidth=2, label='Distribución KDE')
        
        ax.set_title(f'{variable}\nMedia: {media:.2f}, σ: {desv_std:.2f}', fontsize=12)
        ax.set_xlabel(variable)
        ax.set_ylabel('Frecuencia')
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================
# 10. VISUALIZACIONES - GRÁFICOS QQ (QUANTILE-QUANTILE)
# =============================================

print("\n" + "=" * 80)
print("GRÁFICOS QQ - ANÁLISIS DE NORMALIDAD")
print("=" * 80)

# Gráficos QQ para evaluar normalidad
variables_qq = ['PTS', 'AST', 'REB', 'FG_PCT']

fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for i, variable in enumerate(variables_qq):
    if i < len(axes) and variable in df.columns:
        ax = axes[i]
        valores = df[variable].dropna()
        
        # Gráfico QQ
        stats.probplot(valores, dist="norm", plot=ax)
        ax.set_title(f'Gráfico QQ - {variable}', fontsize=14)
        ax.grid(True, alpha=0.3)
        
        # Calcular estadísticas de normalidad
        stat_sw, p_sw = stats.shapiro(valores)
        stat_ks, p_ks = stats.kstest(valores, 'norm', 
                                     args=(valores.mean(), valores.std()))
        
        # Añadir texto con resultados de pruebas
        texto = f"Shapiro-Wilk: p = {p_sw:.4f}\n"
        texto += f"KS Test: p = {p_ks:.4f}\n"
        texto += "Normalidad" if p_sw > 0.05 else "No normal"
        
        ax.text(0.05, 0.95, texto, transform=ax.transAxes, fontsize=10,
                verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

plt.tight_layout()
plt.show()

# =============================================
# 11. VISUALIZACIONES - BOXPLOTS COMPARATIVOS
# =============================================

print("\n" + "=" * 80)
print("BOXPLOTS COMPARATIVOS CON MEDIDAS ESTADÍSTICAS")
print("=" * 80)

# Boxplots con medidas estadísticas detalladas
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.flatten()

variables_boxplot = ['PTS', 'AST', 'REB', 'FG_PCT', 'FT_PCT', 'PLUS_MINUS']
variables_boxplot = [v for v in variables_boxplot if v in existing_metrics]

for i, variable in enumerate(variables_boxplot[:6]):
    ax = axes[i]
    valores = df[variable].dropna()
    
    # Crear boxplot
    bp = ax.boxplot(valores, vert=True, patch_artist=True, 
                   showmeans=True, meanline=True,
                   boxprops=dict(facecolor='lightblue', color='darkblue'),
                   medianprops=dict(color='red', linewidth=2),
                   meanprops=dict(color='green', linewidth=2),
                   whiskerprops=dict(color='darkblue', linewidth=1.5),
                   capprops=dict(color='darkblue', linewidth=1.5),
                   flierprops=dict(marker='o', markerfacecolor='red', 
                                  markersize=6, alpha=0.6))
    
    # Calcular medidas de dispersión
    Q1 = valores.quantile(0.25)
    Q3 = valores.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = valores[(valores < limite_inferior) | (valores > limite_superior)]
    
    # Añadir texto con estadísticas
    texto = f"N: {len(valores):,}\n"
    texto += f"Media: {valores.mean():.2f}\n"
    texto += f"Mediana: {valores.median():.2f}\n"
    texto += f"Q1: {Q1:.2f}, Q3: {Q3:.2f}\n"
    texto += f"IQR: {IQR:.2f}\n"
    texto += f"Outliers: {len(outliers)} ({len(outliers)/len(valores)*100:.1f}%)"
    
    ax.text(0.02, 0.98, texto, transform=ax.transAxes, fontsize=9,
            verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    ax.set_title(f'Boxplot de {variable}', fontsize=14)
    ax.set_ylabel(variable)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# =============================================
# 12. ANÁLISIS DE ASIMETRÍA Y CURTOSIS
# =============================================

print("\n" + "=" * 80)
print("ANÁLISIS DE ASIMETRÍA Y CURTOSIS")
print("=" * 80)

# Calcular asimetría y curtosis para todas las variables
analisis_forma = []

for variable in existing_metrics[:10]:
    valores = df[variable].dropna()
    
    if len(valores) > 0:
        skewness = valores.skew()
        kurtosis = valores.kurtosis()
        
        # Interpretar asimetría
        if abs(skewness) < 0.5:
            interpretacion_skew = "Simétrica"
        elif skewness > 0.5:
            interpretacion_skew = "Asimetría positiva (cola derecha)"
        else:
            interpretacion_skew = "Asimetría negativa (cola izquierda)"
        
        # Interpretar curtosis
        if abs(kurtosis) < 0.5:
            interpretacion_kurt = "Mesocúrtica (similar a normal)"
        elif kurtosis > 0.5:
            interpretacion_kurt = "Leptocúrtica (picos pronunciados)"
        else:
            interpretacion_kurt = "Platicúrtica (picos aplanados)"
        
        analisis_forma.append({
            'Variable': variable,
            'Asimetría (Skewness)': f"{skewness:.3f}",
            'Interpretación Asimetría': interpretacion_skew,
            'Curtosis': f"{kurtosis:.3f}",
            'Interpretación Curtosis': interpretacion_kurt
        })

df_forma = pd.DataFrame(analisis_forma)
print("\nAnálisis de forma de distribución (Asimetría y Curtosis):")
display(df_forma)

# Visualizar asimetría y curtosis
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Gráfico de asimetría
skew_values = [float(x['Asimetría (Skewness)']) for x in analisis_forma]
variables = [x['Variable'] for x in analisis_forma]

axes[0].bar(variables, skew_values, color=['red' if x > 0.5 else 'blue' if x < -0.5 else 'gray' for x in skew_values])
axes[0].axhline(y=0, color='black', linestyle='-', linewidth=1)
axes[0].axhline(y=0.5, color='red', linestyle='--', linewidth=1, alpha=0.5, label='Límite simetría')
axes[0].axhline(y=-0.5, color='red', linestyle='--', linewidth=1, alpha=0.5)
axes[0].set_title('Asimetría de las Variables', fontsize=14)
axes[0].set_xlabel('Variable')
axes[0].set_ylabel('Coeficiente de Asimetría')
axes[0].set_xticklabels(variables, rotation=45, ha='right')
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# Gráfico de curtosis
kurt_values = [float(x['Curtosis']) for x in analisis_forma]
axes[1].bar(variables, kurt_values, color=['red' if x > 0.5 else 'blue' if x < -0.5 else 'gray' for x in kurt_values])
axes[1].axhline(y=0, color='black', linestyle='-', linewidth=1, label='Normal (0)')
axes[1].axhline(y=0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5, label='Límite curtosis')
axes[1].axhline(y=-0.5, color='orange', linestyle='--', linewidth=1, alpha=0.5)
axes[1].set_title('Curtosis de las Variables', fontsize=14)
axes[1].set_xlabel('Variable')
axes[1].set_ylabel('Curtosis')
axes[1].set_xticklabels(variables, rotation=45, ha='right')
axes[1].legend()
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.show()

# =============================================
# 13. RESUMEN ESTADÍSTICO FINAL
# =============================================

print("\n" + "=" * 80)
print("RESUMEN ESTADÍSTICO FINAL")
print("=" * 80)

# Crear resumen ejecutivo
print("\nRESUMEN EJECUTIVO DEL ANÁLISIS ESTADÍSTICO:")
print("-" * 50)

# Estadísticas generales
print(f"1. DATOS GENERALES:")
print(f"   • Total de observaciones: {df.shape[0]:,}")
print(f"   • Variables analizadas: {len(existing_metrics)}")
print(f"   • Periodo analizado: {df['SEASON_YEAR'].min()} - {df['SEASON_YEAR'].max()}")

# Análisis de las variables más importantes
print(f"\n2. VARIABLES PRINCIPALES:")

for variable in ['PTS', 'AST', 'REB']:
    if variable in df.columns:
        valores = df[variable].dropna()
        print(f"\n   {variable}:")
        print(f"   • Rango: {valores.min():.1f} - {valores.max():.1f}")
        print(f"   • Media ± Desv. Est.: {valores.mean():.2f} ± {valores.std():.2f}")
        print(f"   • Mediana (Q2): {valores.median():.2f}")
        print(f"   • IQR (Q3-Q1): {valores.quantile(0.75) - valores.quantile(0.25):.2f}")
        print(f"   • Coef. Variación: {(valores.std() / valores.mean() * 100):.1f}%")

# Distribución de los datos
print(f"\n3. CARACTERÍSTICAS DE DISTRIBUCIÓN:")

# Analizar normalidad de PTS
if 'PTS' in df.columns:
    pts_valores = df['PTS'].dropna()
    skew_pts = pts_valores.skew()
    
    if abs(skew_pts) < 0.5:
        normalidad = "Distribución aproximadamente normal"
    elif skew_pts > 0:
        normalidad = "Distribución con asimetría positiva (sesgo a la derecha)"
    else:
        normalidad = "Distribución con asimetría negativa (sesgo a la izquierda)"
    
    print(f"   • Puntos (PTS): {normalidad} (Skewness: {skew_pts:.3f})")

# Valores atípicos
print(f"\n4. VALORES ATÍPICOS (OUTLIERS):")

for variable in ['PTS', 'AST', 'REB']:
    if variable in df.columns:
        valores = df[variable].dropna()
        Q1 = valores.quantile(0.25)
        Q3 = valores.quantile(0.75)
        IQR = Q3 - Q1
        limite_inferior = Q1 - 1.5 * IQR
        limite_superior = Q3 + 1.5 * IQR
        outliers = valores[(valores < limite_inferior) | (valores > limite_superior)]
        
        print(f"   • {variable}: {len(outliers):,} outliers ({len(outliers)/len(valores)*100:.1f}%)")

# =============================================
# 14. EXPORTACIÓN DE RESULTADOS
# =============================================

print("\n" + "=" * 80)
print("EXPORTACIÓN DE RESULTADOS")
print("=" * 80)

# Exportar DataFrame con medidas estadísticas
df_estadisticas.to_csv('medidas_estadisticas_nba.csv', index=False, encoding='utf-8-sig')
print(f"✓ Medidas estadísticas exportadas a 'medidas_estadisticas_nba.csv'")

# Exportar tablas de frecuencia para variables principales
for variable in variables_tabla[:3]:
    resultado = crear_tabla_frecuencias(df, variable, n_bins=8)
    if resultado:
        tabla, _ = resultado
        nombre_archivo = f'tabla_frecuencias_{variable}.csv'
        tabla.to_csv(nombre_archivo, index=False, encoding='utf-8-sig')
        print(f"✓ Tabla de frecuencias de {variable} exportada a '{nombre_archivo}'")

# Crear y exportar reporte resumido
reporte_resumen = {
    'Fecha_Generacion': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
    'Total_Observaciones': df.shape[0],
    'Total_Variables': df.shape[1],
    'Variables_Analizadas': len(existing_metrics),
    'Periodo_Analizado': f"{df['SEASON_YEAR'].min()} - {df['SEASON_YEAR'].max()}"
}

# Agregar estadísticas clave
if 'PTS' in df.columns:
    pts_valores = df['PTS'].dropna()
    reporte_resumen['PTS_Media'] = pts_valores.mean()
    reporte_resumen['PTS_DesvEst'] = pts_valores.std()
    reporte_resumen['PTS_Mediana'] = pts_valores.median()

df_reporte = pd.DataFrame([reporte_resumen])
df_reporte.to_csv('reporte_analisis_nba.csv', index=False, encoding='utf-8-sig')
print(f"✓ Reporte de análisis exportado a 'reporte_analisis_nba.csv'")

print("\n" + "=" * 80)
print("ANÁLISIS COMPLETADO EXITOSAMENTE")
print("=" * 80)
print("\nArchivos generados:")
print("1. medidas_estadisticas_nba.csv - Todas las medidas estadísticas")
print("2. tabla_frecuencias_[VARIABLE].csv - Tablas de frecuencia")
print("3. reporte_analisis_nba.csv - Reporte resumido")
print("4. nba_data_cleaned.csv - Dataset limpio (si se ejecutó la sección 15)")

# =============================================
# 15. FUNCIONES UTILITARIAS ADICIONALES
# =============================================

def analisis_variable_detallado(data, variable):
    """Función para análisis detallado de una variable específica"""
    
    print(f"\n{'='*60}")
    print(f"ANÁLISIS DETALLADO DE: {variable}")
    print(f"{'='*60}")
    
    if variable not in data.columns:
        print(f"Variable '{variable}' no encontrada en el dataset")
        return
    
    valores = data[variable].dropna()
    
    if len(valores) == 0:
        print(f"No hay datos válidos para la variable '{variable}'")
        return
    
    # 1. Medidas de tendencia central
    print("\n1. MEDIDAS DE TENDENCIA CENTRAL:")
    print(f"   • Media: {valores.mean():.4f}")
    print(f"   • Mediana: {valores.median():.4f}")
    moda = valores.mode()
    if not moda.empty:
        print(f"   • Moda: {moda.iloc[0]:.4f}" + 
              (f" (frecuencia: {len(valores[valores == moda.iloc[0]])})" if len(valores) > 0 else ""))
    
    # 2. Medidas de posición
    print("\n2. MEDIDAS DE POSICIÓN:")
    percentiles = [10, 25, 50, 75, 90]
    for p in percentiles:
        print(f"   • P{p}: {np.percentile(valores, p):.4f}")
    
    # 3. Medidas de dispersión
    print("\n3. MEDIDAS DE DISPERSIÓN:")
    print(f"   • Varianza: {valores.var():.4f}")
    print(f"   • Desviación Estándar: {valores.std():.4f}")
    print(f"   • Rango: {valores.max() - valores.min():.4f}")
    print(f"   • IQR: {valores.quantile(0.75) - valores.quantile(0.25):.4f}")
    print(f"   • Coef. Variación: {(valores.std() / valores.mean() * 100):.2f}%")
    
    # 4. Medidas de forma
    print("\n4. MEDIDAS DE FORMA:")
    print(f"   • Asimetría: {valores.skew():.4f}")
    print(f"   • Curtosis: {valores.kurtosis():.4f}")
    
    # 5. Análisis de outliers
    print("\n5. ANÁLISIS DE OUTLIERS:")
    Q1 = valores.quantile(0.25)
    Q3 = valores.quantile(0.75)
    IQR = Q3 - Q1
    limite_inferior = Q1 - 1.5 * IQR
    limite_superior = Q3 + 1.5 * IQR
    outliers = valores[(valores < limite_inferior) | (valores > limite_superior)]
    
    print(f"   • Límite inferior: {limite_inferior:.4f}")
    print(f"   • Límite superior: {limite_superior:.4f}")
    print(f"   • Número de outliers: {len(outliers):,}")
    print(f"   • Porcentaje de outliers: {len(outliers)/len(valores)*100:.2f}%")
    
    # 6. Visualización
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Histograma
    axes[0].hist(valores, bins=20, edgecolor='black', alpha=0.7)
    axes[0].set_title(f'Distribución de {variable}')
    axes[0].set_xlabel(variable)
    axes[0].set_ylabel('Frecuencia')
    
    # Boxplot
    axes[1].boxplot(valores, vert=True, patch_artist=True)
    axes[1].set_title(f'Boxplot de {variable}')
    axes[1].set_ylabel(variable)
    
    # Gráfico QQ
    stats.probplot(valores, dist="norm", plot=axes[2])
    axes[2].set_title(f'Gráfico QQ de {variable}')
    
    plt.tight_layout()
    plt.show()

# Ejemplo de uso de la función
print("\n" + "=" * 80)
print("FUNCIÓN DE ANÁLISIS DETALLADO DISPONIBLE")
print("=" * 80)
print("Para analizar cualquier variable en detalle, use:")
print("analisis_variable_detallado(df, 'NOMBRE_VARIABLE')")
print("\nEjemplo: analisis_variable_detallado(df, 'PTS')")


# =============================================
# 16. TRANSFORMACIONES COMPLETAS
# =============================================


print("\n" + "="*80)
print("TRANSFORMACIONES COMPLETAS")
print("="*80)

# Cargar datos (si no están ya cargados)
# df = pd.read_csv('regular_season_totals_2010_2024.csv')

print(f"\n📊 TAMAÑO DEL DATASET: {df.shape[0]:,} filas × {df.shape[1]} columnas")

if df.shape[0] < 10000:
    print("⚠️  ADVERTENCIA: Dataset más pequeño de lo esperado")

# 1. IMPUTACIÓN DE VALORES FALTANTES 
print("\n1. IMPUTACIÓN DE VALORES FALTANTES")

valores_faltantes = {}
for var in ['PLUS_MINUS', 'FG_PCT', 'FG3_PCT', 'FT_PCT', 'REB']:
    if var in df.columns:
        n_missing = df[var].isnull().sum()
        pct_missing = n_missing / len(df) * 100
        valores_faltantes[var] = (n_missing, pct_missing)
        
        if n_missing > 0:
            median_val = df[var].median()
            df[var].fillna(median_val, inplace=True)
            print(f"   ✓ {var}: {n_missing} valores ({pct_missing:.2f}%) imputados con mediana={median_val:.3f}")

if 'REB' in df.columns and all(col in df.columns for col in ['OREB', 'DREB']):
    df['REB_CALCULADO'] = df['OREB'] + df['DREB']
    print(f"   ✓ REB recalculado como OREB + DREB")

# 2. CODIFICACIÓN CATEGÓRICA COMPLETA
print("\n2. CODIFICACIÓN CATEGÓRICA COMPLETA")

if 'WL' in df.columns:
    df['WL_BINARY'] = df['WL'].map({'W': 1, 'L': 0})
    print(f"   ✓ WL codificado a binario: W=1, L=0")

if 'AVAILABLE_FLAG' in df.columns:
    df['AVAILABLE_BINARY'] = df['AVAILABLE_FLAG'].map({'Available': 1, 'Not Available': 0})
    print(f"   ✓ AVAILABLE_FLAG codificado a binario")

if 'MATCHUP' in df.columns:
    df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 1 if '@' not in str(x) else 0)
    print(f"   ✓ IS_HOME extraído de MATCHUP: Home=1, Away=0")

if 'TEAM_ABBREVIATION' in df.columns:
    team_dummies = pd.get_dummies(df['TEAM_ABBREVIATION'], prefix='TEAM', drop_first=True)
    df = pd.concat([df, team_dummies], axis=1)
    print(f"   ✓ One-Hot Encoding para {df['TEAM_ABBREVIATION'].nunique()} equipos")

if 'SEASON_YEAR' in df.columns:
    df['SEASON_NUM'] = df['SEASON_YEAR'].astype(str).str.extract(r'(\d{4})').astype(float) - 2009
    print(f"   ✓ SEASON_NUM creado: SEASON_YEAR - 2009")

if 'GAME_DATE' in df.columns:
    df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
    df['MONTH'] = df['GAME_DATE'].dt.month
    df['DAY_OF_WEEK'] = df['GAME_DATE'].dt.day_name()
    print(f"   ✓ Variables temporales extraídas: MONTH, DAY_OF_WEEK")

if 'MATCHUP' in df.columns:
    def clasificar_partido(matchup):
        if 'vs.' in matchup:
            return 'HOME'
        elif '@' in matchup:
            return 'AWAY'
        else:
            return 'UNKNOWN'
    
    df['GAME_TYPE'] = df['MATCHUP'].apply(clasificar_partido)
    print(f"   ✓ GAME_TYPE creado de MATCHUP")

# 2.6 CREACIÓN DE SEASON_PART
if 'MONTH' in df.columns:
    def get_season_part(month):
        if month in [10, 11, 12]:  # Oct-Dic (inicio temporada)
            return 1
        elif month in [1, 2, 3]:   # Ene-Mar (mitad temporada)
            return 2
        else:                       # Abr-Jun (final temporada)
            return 3
    
    df['SEASON_PART'] = df['MONTH'].apply(get_season_part)
    print(f"   ✓ SEASON_PART creado: 1=Oct-Dic, 2=Ene-Mar, 3=Abr-Jun")

# 3. CREACIÓN DE VARIABLES DERIVADAS
print("\n3. CREACIÓN DE VARIABLES DERIVADAS COMPLETAS")

if all(col in df.columns for col in ['FGA', 'OREB', 'TOV', 'FTA']):
    df['POSS'] = df['FGA'] - df['OREB'] + df['TOV'] + 0.4 * df['FTA']
    print(f"   ✓ POSS (posesiones) calculado")

if all(col in df.columns for col in ['POSS', 'MIN']):
    df['PACE'] = df['POSS'] / df['MIN'] * 48
    print(f"   ✓ PACE (ritmo) calculado: POSS/MIN × 48")

if all(col in df.columns for col in ['PTS', 'POSS']):
    df['OFF_EFF'] = df['PTS'] / df['POSS'] * 100
    print(f"   ✓ OFF_EFF (eficiencia ofensiva) calculada: PTS/POSS × 100")

if all(col in df.columns for col in ['AST', 'TOV']):
    df['AST_TO_RATIO'] = df['AST'] / df['TOV'].replace(0, np.nan)
    print(f"   ✓ AST_TO_RATIO calculado: AST/TOV")

if all(col in df.columns for col in ['FG3A', 'FGA']):
    df['FG3A_RATE'] = df['FG3A'] / df['FGA']
    print(f"   ✓ FG3A_RATE calculado: FG3A/FGA")

if all(col in df.columns for col in ['FG_PCT', 'FG3_PCT']):
    df['EFFICIENT_SCORING'] = df['FG_PCT'] + 0.5 * df['FG3_PCT']
    print(f"   ✓ EFFICIENT_SCORING calculado: FG_PCT + 0.5×FG3_PCT")

# 4. TRANSFORMACIONES PARA NORMALIDAD 
print("\n4. TRANSFORMACIONES PARA NORMALIDAD")

if 'PLUS_MINUS' in df.columns:
    min_val = df['PLUS_MINUS'].min()
    shift_value = abs(min_val) + 1 if min_val <= 0 else 0
    
    if shift_value > 0:
        df['PLUS_MINUS_POS'] = df['PLUS_MINUS'] + shift_value
        df['PLUS_MINUS_BOXCOX'] = (df['PLUS_MINUS_POS']**0.5 - 1) / 0.5
        print(f"   ✓ PLUS_MINUS transformado con Box-Cox (λ=0.5)")

log_vars = ['FG3M', 'FGM', 'PTS', 'AST', 'REB']
for var in log_vars:
    if var in df.columns and df[var].min() >= 0:
        df[f'{var}_LOG'] = np.log1p(df[var])
print(f"   ✓ Transformación log aplicada a {len(log_vars)} variables")

# 5. ESTANDARIZACIÓN Y NORMALIZACIÓN
print("\n5. ESTANDARIZACIÓN Y NORMALIZACIÓN")

z_vars = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV']
z_vars = [v for v in z_vars if v in df.columns]

if z_vars:
    scaler_z = StandardScaler()
    z_data = scaler_z.fit_transform(df[z_vars])
    z_df = pd.DataFrame(z_data, columns=[f'{v}_Z' for v in z_vars])
    df = pd.concat([df, z_df], axis=1)
    
    # Crear las columnas estandarizadas con nombres específicos
    std_mapping = {
        'PTS_Z': 'PTS_STD',
        'AST_Z': 'AST_STD',
        'REB_Z': 'REB_STD', 
        'STL_Z': 'STL_STD',
        'BLK_Z': 'BLK_STD',
        'TOV_Z': 'TOV_STD'
    }
    
    for z_col, std_col in std_mapping.items():
        if z_col in df.columns:
            df[std_col] = df[z_col]
    
    print(f"   ✓ {len(z_vars)} variables estandarizadas (Z-score)")
    print(f"   ✓ Columnas _STD creadas: AST_STD, REB_STD, STL_STD, BLK_STD, TOV_STD, PTS_STD")

mm_vars = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
mm_vars = [v for v in mm_vars if v in df.columns]

if mm_vars:
    scaler_mm = MinMaxScaler()
    mm_data = scaler_mm.fit_transform(df[mm_vars])
    mm_df = pd.DataFrame(mm_data, columns=[f'{v}_MM' for v in mm_vars])
    df = pd.concat([df, mm_df], axis=1)
    print(f"   ✓ {len(mm_vars)} porcentajes normalizados [0,1]")

# 6. TRATAMIENTO DE OUTLIERS
print("\n6. TRATAMIENTO DE OUTLIERS")

if 'PLUS_MINUS' in df.columns:
    df['PLUS_MINUS_WINS'] = winsorize(df['PLUS_MINUS'], limits=[0.05, 0.05])
    print(f"   ✓ PLUS_MINUS winsorizado al 5% y 95%")

# 7. VERIFICACIONES POST-TRANSFORMACIÓN 
print("\n7. VERIFICACIONES POST-TRANSFORMACIÓN")

test_vars = ['PTS', 'PLUS_MINUS']
for var in test_vars:
    if var in df.columns:
        data = df[var].dropna()
        if len(data) > 5000:
            stat, p = stats.kstest((data - data.mean())/data.std(), 'norm')
            test_name = "Kolmogorov-Smirnov"
        else:
            stat, p = stats.shapiro(data)
            test_name = "Shapiro-Wilk"
        
        normal = '(Normal)' if p > 0.05 else '(No normal)'
        print(f"   • {var}: {test_name} p={p:.4f} {normal}")

if 'SEASON_NUM' in df.columns and 'PTS' in df.columns:
    groups = [df[df['SEASON_NUM'] == season]['PTS'].values 
              for season in sorted(df['SEASON_NUM'].unique())]
    
    if len(groups) > 1:
        stat, p = levene(*groups)
        print(f"   • Levene test (PTS por temporada): p={p:.4f}")

vif_vars = ['PTS', 'AST', 'REB', 'FG_PCT', 'FG3_PCT']
vif_vars = [v for v in vif_vars if v in df.columns]

if len(vif_vars) > 1:
    from statsmodels.tools.tools import add_constant
    X = add_constant(df[vif_vars].dropna())
    
    vif_data = []
    for i, col in enumerate(X.columns):
        if col != 'const':
            vif = variance_inflation_factor(X.values, i)
            vif_data.append({'Variable': col, 'VIF': vif})
    
    vif_df = pd.DataFrame(vif_data)
    vif_promedio = vif_df['VIF'].mean()
    print(f"   • VIF promedio: {vif_promedio:.2f}")

print("\n8. VERIFICACIÓN DE COLUMNAS CREADAS")
new_columns = ['SEASON_PART', 'FG3A_RATE', 'AST_STD', 'REB_STD', 'STL_STD', 'BLK_STD', 'TOV_STD', 
               'PTS_STD', 'OFF_EFF', 'AST_TO_RATIO']
for col in new_columns:
    if col in df.columns:
        print(f"   ✓ {col}: creada exitosamente")
    else:
        print(f"   ✗ {col}: NO creada")

# 9. CREACIÓN DE VERSIONES DEL DATASET 
print("\n9. CREACIÓN DE VERSIONES DEL DATASET")

cols_raw = [c for c in df.columns if not any(x in c for x in ['_Z', '_MM', '_LOG', '_BOXCOX', '_WINS', '_BINARY', '_STD', '_RATE'])]
df_raw = df[cols_raw].copy()
df_raw.to_csv('nba_raw.csv', index=False)
print(f"   ✓ nba_raw.csv: {df_raw.shape[1]} columnas")

cols_std = [c for c in df.columns if any(x in c for x in ['_Z', '_STD', 'IS_HOME', 'WL_BINARY', 'SEASON_NUM'])]
df_std = df[cols_std].copy()
df_std.to_csv('nba_standardized.csv', index=False)
print(f"   ✓ nba_standardized.csv: {df_std.shape[1]} columnas")

cols_norm = [c for c in df.columns if any(x in c for x in ['_MM', 'IS_HOME', 'WL_BINARY', 'SEASON_NUM'])]
df_norm = df[cols_norm].copy()
df_norm.to_csv('nba_normalized.csv', index=False)
print(f"   ✓ nba_normalized.csv: {df_norm.shape[1]} columnas")

cols_trans = [c for c in df.columns if any(x in c for x in [
    '_LOG', '_BOXCOX', '_STD', '_MM', '_Z', '_BINARY', '_RATE', 
    'RATIO', 'EFF', 'PACE', 'POSS', 'IS_HOME', 'SEASON_NUM', 'SEASON_PART',
    'FG3A_RATE', 'AST_STD', 'REB_STD', 'STL_STD', 'BLK_STD', 'TOV_STD', 'PTS_STD'
]) or c in ['OFF_EFF', 'AST_TO_RATIO', 'WL_BINARY', 'AVAILABLE_BINARY']]

df_trans = df[cols_trans].copy()
df_trans.to_csv('nba_transformed.csv', index=False)
print(f"   ✓ nba_transformed.csv: {df_trans.shape[1]} columnas")

df.to_csv('nba_complete_transformed.csv', index=False)

print(f"\n{'='*80}")
print("TRANSFORMACIONES COMPLETADAS EXITOSAMENTE")
print("="*80)
print(f"\nResumen final:")
print(f"• Dataset original: {df.shape[0]:,} filas × {df.shape[1]} columnas")
print(f"• Nuevas columnas categóricas: SEASON_PART")
print(f"• Nuevas columnas derivadas: OFF_EFF, AST_TO_RATIO, FG3A_RATE")
print(f"• Nuevas columnas estandarizadas: AST_STD, REB_STD, STL_STD, BLK_STD, TOV_STD, PTS_STD")


# =============================================
# CONFIGURACIÓN INICIAL - TECNICAS ESTADISTICAS 
# =============================================


warnings.filterwarnings('ignore')

# Configuración estética
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)

# Cargar el dataset transformado (ASEGÚRATE DE TENERLO)
try:
    df = pd.read_csv('nba_transformed.csv')
    print(f"✓ Dataset transformado cargado: {df.shape[0]:,} filas × {df.shape[1]} columnas")
except:
    print("❌ ERROR: Primero genera nba_transformed.csv con el código anterior")
    print("   Ejecuta la sección de transformaciones completas")
    exit()

# Verificar columnas esenciales
essential_cols = ['WL_BINARY', 'FG3M_LOG', 'FG_PCT_MM', 'PTS_LOG', 
                  'SEASON_NUM', 'FG3A_RATE', 'PLUS_MINUS_BOXCOX', 
                  'AST_STD', 'REB_STD', 'STL_STD', 'BLK_STD', 'TOV_STD']
missing = [col for col in essential_cols if col not in df.columns]
if missing:
    print(f"❌ Faltan columnas: {missing}")
    print("   Asegúrate de que el dataset transformado incluya todas las transformaciones")
else:
    print("✓ Todas las columnas esenciales disponibles")
    
# =============================================
# PREGUNTA 1: TRIPLES VS VICTORIA (REGRESIÓN LOGÍSTICA)
# =============================================

print("\n" + "="*80)
print("PREGUNTA 1: ¿EL NÚMERO DE TRIPLES PREDICE LA VICTORIA?")
print("="*80)

# ----------------------------------------------------
# 1. PREPARACIÓN DE DATOS
# ----------------------------------------------------
print("\n1. PREPARACIÓN DE DATOS PARA REGRESIÓN LOGÍSTICA")

# Variables predictoras: Triples y porcentaje de tiro (control)
X = df[['FG3M_LOG', 'FG_PCT_MM']].copy()

# Añadir ventaja local como variable de control si existe
if 'IS_HOME' in df.columns:
    X['IS_HOME'] = df['IS_HOME']
    print("   • Variable de control incluida: IS_HOME (ventaja local)")

# Variable respuesta: Victoria (1) o Derrota (0)
y = df['WL_BINARY']

print(f"   • Predictores: {list(X.columns)}")
print(f"   • Variable respuesta: WL_BINARY (1=Victoria, 0=Derrota)")
print(f"   • Muestra total: {len(X):,} partidos")
print(f"   • Victorias: {y.sum():,} ({y.mean()*100:.1f}%)")
print(f"   • Derrotas: {len(y)-y.sum():,} ({(1-y.mean())*100:.1f}%)")

# ----------------------------------------------------
# 2. DIVISIÓN EN CONJUNTOS DE ENTRENAMIENTO Y PRUEBA
# ----------------------------------------------------
from sklearn.model_selection import train_test_split

# 70% entrenamiento, 30% prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

print(f"\n2. DIVISIÓN DE DATOS:")
print(f"   • Entrenamiento: {len(X_train):,} partidos ({len(X_train)/len(X)*100:.0f}%)")
print(f"   • Prueba: {len(X_test):,} partidos ({len(X_test)/len(X)*100:.0f}%)")

# ----------------------------------------------------
# 3. ENTRENAMIENTO DEL MODELO DE REGRESIÓN LOGÍSTICA
# ----------------------------------------------------
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, roc_curve

print("\n3. ENTRENAMIENTO DEL MODELO DE REGRESIÓN LOGÍSTICA")

# Crear y entrenar el modelo
log_model = LogisticRegression(
    random_state=42,
    max_iter=1000,
    class_weight='balanced'  # Manejo de desbalance de clases
)
log_model.fit(X_train, y_train)

# Coeficientes del modelo
coefficients = pd.DataFrame({
    'Variable': ['Intercept'] + list(X.columns),
    'Coeficiente': [log_model.intercept_[0]] + list(log_model.coef_[0]),
    'Odds_Ratio': [np.exp(log_model.intercept_[0])] + list(np.exp(log_model.coef_[0]))
})

print("\n   COEFICIENTES DEL MODELO:")
print("   " + "-"*50)
for _, row in coefficients.iterrows():
    if row['Variable'] == 'Intercept':
        print(f"   {row['Variable']:15s}: {row['Coeficiente']:7.4f} (OR: {row['Odds_Ratio']:.4f})")
    else:
        sign = "↑" if row['Coeficiente'] > 0 else "↓"
        print(f"   {row['Variable']:15s}: {row['Coeficiente']:7.4f} (OR: {row['Odds_Ratio']:.4f}) {sign}")

# ----------------------------------------------------
# 4. EVALUACIÓN DEL MODELO
# ----------------------------------------------------
print("\n4. EVALUACIÓN DEL MODELO EN DATOS DE PRUEBA")

# Predicciones
y_pred = log_model.predict(X_test)
y_pred_proba = log_model.predict_proba(X_test)[:, 1]

# Métricas de evaluación
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
roc_auc = roc_auc_score(y_test, y_pred_proba)

print(f"   • Exactitud (Accuracy): {accuracy:.3f}")
print(f"   • Precisión: {precision:.3f} (de los predichos como victorias, cuántos realmente ganaron)")
print(f"   • Sensibilidad (Recall): {recall:.3f} (de las verdaderas victorias, cuántas detectamos)")
print(f"   • F1-Score: {f1:.3f} (media armónica de precisión y sensibilidad)")
print(f"   • AUC-ROC: {roc_auc:.3f} (capacidad discriminativa)")

# Matriz de confusión
cm = confusion_matrix(y_test, y_pred)
cm_df = pd.DataFrame(cm, 
                     index=['Derrota Real', 'Victoria Real'], 
                     columns=['Derrota Predicha', 'Victoria Predicha'])

print("\n   MATRIZ DE CONFUSIÓN:")
print("   " + "-"*35)
print(f"   {'':20s} | {'Predicción':^20s}")
print(f"   {'':20s} | {'Derrota':^10s} | {'Victoria':^10s}")
print("   " + "-"*35)
print(f"   {'Derrota Real':20s} | {cm[0,0]:^10d} | {cm[0,1]:^10d}")
print(f"   {'Victoria Real':20s} | {cm[1,0]:^10d} | {cm[1,1]:^10d}")
print("   " + "-"*35)

# ----------------------------------------------------
# 5. ANÁLISIS DE SIGNIFICANCIA ESTADÍSTICA
# ----------------------------------------------------
print("\n5. ANÁLISIS DE SIGNIFICANCIA ESTADÍSTICA")

# Usando statsmodels para obtener p-valores
import statsmodels.api as sm

# Añadir constante para el intercepto
X_sm = sm.add_constant(X_train)
logit_model = sm.Logit(y_train, X_sm)
result = logit_model.fit(disp=0, maxiter=1000)

# Resumen estadístico
print("\n   RESUMEN ESTADÍSTICO DEL MODELO:")
print("   " + "-"*60)
print("   Variable        Coef.   Error Est.   z      P>|z|   [0.025   0.975]")
print("   " + "-"*60)

for i, var in enumerate(['const'] + list(X.columns)):
    coef = result.params[i]
    std_err = result.bse[i]
    z = result.tvalues[i]
    p_val = result.pvalues[i]
    ci_low = result.conf_int()[0][i]
    ci_high = result.conf_int()[1][i]
    
    signif = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
    
    print(f"   {var:15s} {coef:7.4f} {std_err:11.4f} {z:6.2f} {p_val:7.4f} {ci_low:7.4f} {ci_high:7.4f} {signif}")

print("   " + "-"*60)
print("   *** p<0.001, ** p<0.01, * p<0.05")

# ----------------------------------------------------
# 6. VISUALIZACIONES
# ----------------------------------------------------
print("\n6. VISUALIZACIÓN DE RESULTADOS")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1 Curva ROC
fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
axes[0, 0].plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.3f})')
axes[0, 0].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Aleatorio')
axes[0, 0].set_xlabel('Tasa de Falsos Positivos (1 - Especificidad)')
axes[0, 0].set_ylabel('Tasa de Verdaderos Positivos (Sensibilidad)')
axes[0, 0].set_title('Curva ROC - Capacidad Predictiva del Modelo')
axes[0, 0].legend(loc="lower right")
axes[0, 0].grid(True, alpha=0.3)

# 6.2 Odds Ratios
variables = coefficients['Variable'][1:]
odds_ratios = coefficients['Odds_Ratio'][1:]
colors = ['green' if or_val > 1 else 'red' for or_val in odds_ratios]
axes[0, 1].barh(variables, odds_ratios, color=colors)
axes[0, 1].axvline(x=1, color='black', linestyle='--', alpha=0.5)
axes[0, 1].set_xlabel('Odds Ratio')
axes[0, 1].set_title('Efecto de Variables en Probabilidad de Victoria')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# 6.3 Probabilidad predicha vs triples
axes[1, 0].scatter(df['FG3M_LOG'], df['WL_BINARY'], alpha=0.1, color='gray', label='Datos reales')
sorted_idx = np.argsort(X_test['FG3M_LOG'])
axes[1, 0].plot(X_test['FG3M_LOG'].iloc[sorted_idx], 
                y_pred_proba[sorted_idx], 
                color='red', linewidth=2, label='Probabilidad predicha')
axes[1, 0].set_xlabel('Triples Anotados (log-transformados)')
axes[1, 0].set_ylabel('Probabilidad de Victoria')
axes[1, 0].set_title('Relación entre Triples y Probabilidad de Ganar')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 6.4 Matriz de confusión heatmap
sns.heatmap(cm_df, annot=True, fmt='d', cmap='Blues', ax=axes[1, 1])
axes[1, 1].set_title('Matriz de Confusión - Desempeño Predictivo')
axes[1, 1].set_ylabel('Valor Real')
axes[1, 1].set_xlabel('Predicción del Modelo')

plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 7. INTERPRETACIÓN EN CONTEXTO NBA
# ----------------------------------------------------
print("\n" + "="*80)
print("INTERPRETACIÓN DE RESULTADOS - CONTEXTO NBA")
print("="*80)

# Calcular efecto práctico
print("\nEFECTO PRÁCTICO DE LOS TRIPLES EN LA NBA:")
print("-"*50)

# Simular escenarios
scenario_base = pd.DataFrame({
    'FG3M_LOG': [np.log1p(10)],  # 10 triples (≈ valor promedio)
    'FG_PCT_MM': [0.5],          # Eficiencia media
    'IS_HOME': [1]               # Juego local
})

prob_base = log_model.predict_proba(scenario_base)[0, 1]

# Escenario con más triples
scenario_more = scenario_base.copy()
scenario_more['FG3M_LOG'] = np.log1p(15)  # 15 triples
prob_more = log_model.predict_proba(scenario_more)[0, 1]

# Escenario con menos triples
scenario_less = scenario_base.copy()
scenario_less['FG3M_LOG'] = np.log1p(5)   # 5 triples
prob_less = log_model.predict_proba(scenario_less)[0, 1]

print(f"1. Equipo LOCAL con:")
print(f"   • 10 triples + 50% efectividad → {prob_base:.1%} probabilidad de ganar")
print(f"   • 15 triples + 50% efectividad → {prob_more:.1%} probabilidad de ganar")
print(f"   • 5 triples + 50% efectividad → {prob_less:.1%} probabilidad de ganar")
print(f"   → Aumentar de 5 a 15 triples: ▲ {(prob_more-prob_less)*100:.1f} puntos porcentuales")

# Efecto de la eficiencia
scenario_low_eff = scenario_base.copy()
scenario_low_eff['FG_PCT_MM'] = 0.3
prob_low_eff = log_model.predict_proba(scenario_low_eff)[0, 1]

print(f"\n2. Mismos 10 triples pero:")
print(f"   • Con 50% efectividad → {prob_base:.1%} probabilidad")
print(f"   • Con 30% efectividad → {prob_low_eff:.1%} probabilidad")
print(f"   → Efectividad importa MÁS que volumen de triples")

# =============================================
# PREGUNTA 2: EVOLUCIÓN TEMPORAL (ANOVA Y REGRESIÓN)
# =============================================

print("\n" + "="*80)
print("PREGUNTA 2: ¿CÓMO HA EVOLUCIONADO LA NBA OFENSIVA?")
print("="*80)

# ----------------------------------------------------
# 1. ANÁLISIS DESCRIPTIVO POR TEMPORADA
# ----------------------------------------------------
print("\n1. EVOLUCIÓN DESCRIPTIVA POR TEMPORADA")

# Agrupar por temporada
df_season = df.groupby('SEASON_NUM').agg({
    'PTS_LOG': 'mean',
    'FG3A_RATE': 'mean',
    'WL_BINARY': 'mean'
}).reset_index()

# Convertir logaritmos a valores originales (aproximados)
df_season['PTS_APPROX'] = np.expm1(df_season['PTS_LOG'])
df_season['WIN_RATE'] = df_season['WL_BINARY'] * 100

print("\n   ESTADÍSTICAS POR TEMPORADA:")
print("   " + "-"*70)
print("   Temp.  Año NBA    Puntos   Tasa Triples  % Victorias")
print("   " + "-"*70)

for _, row in df_season.iterrows():
    season_year = 2009 + row['SEASON_NUM']
    print(f"   {row['SEASON_NUM']:2.0f}     {season_year}-{season_year+1}   "
          f"{row['PTS_APPROX']:7.1f}     {row['FG3A_RATE']:6.3f}      {row['WIN_RATE']:5.1f}%")

print("   " + "-"*70)

# ----------------------------------------------------
# 2. ANOVA - ¿HAY DIFERENCIAS SIGNIFICATIVAS ENTRE TEMPORADAS?
# ----------------------------------------------------
print("\n2. ANOVA: DIFERENCIAS ENTRE TEMPORADAS")

from scipy.stats import f_oneway

# Preparar datos para ANOVA
pts_by_season = [df[df['SEASON_NUM'] == season]['PTS_LOG'].values 
                 for season in sorted(df['SEASON_NUM'].unique())]

# ANOVA para puntos
f_stat_pts, p_val_pts = f_oneway(*pts_by_season)

# ANOVA para tasa de triples
triples_by_season = [df[df['SEASON_NUM'] == season]['FG3A_RATE'].values 
                     for season in sorted(df['SEASON_NUM'].unique())]
f_stat_3pt, p_val_3pt = f_oneway(*triples_by_season)

print(f"\n   ANOVA PARA PUNTOS POR PARTIDO:")
print(f"   • F-statistic: {f_stat_pts:.2f}")
print(f"   • p-value: {p_val_pts:.6f}")
print(f"   • Conclusión: {'HAY diferencias significativas' if p_val_pts < 0.05 else 'NO hay diferencias significativas'}")

print(f"\n   ANOVA PARA TASA DE TRIPLES:")
print(f"   • F-statistic: {f_stat_3pt:.2f}")
print(f"   • p-value: {p_val_3pt:.6f}")
print(f"   • Conclusión: {'HAY diferencias significativas' if p_val_3pt < 0.05 else 'NO hay diferencias significativas'}")

# ----------------------------------------------------
# 3. TEST DE TUKEY (POST-HOC) PARA IDENTIFICAR DIFERENCIAS ESPECÍFICAS
# ----------------------------------------------------
print("\n3. TEST DE TUKEY: COMPARACIONES MÚLTIPLES")

from statsmodels.stats.multicomp import pairwise_tukeyhsd

# Test de Tukey para puntos
tukey_pts = pairwise_tukeyhsd(
    endog=df['PTS_LOG'],
    groups=df['SEASON_NUM'],
    alpha=0.05
)

print("\n   TUKEY TEST - PUNTOS (diferencias significativas):")
print("   " + "-"*70)

tukey_significant = False
for i in range(len(tukey_pts.groupsunique)):
    for j in range(i+1, len(tukey_pts.groupsunique)):
        idx = i * len(tukey_pts.groupsunique) + j - (i+1)*(i+2)//2
        if tukey_pts.reject[idx]:
            season1 = tukey_pts.groupsunique[i]
            season2 = tukey_pts.groupsunique[j]
            mean_diff = tukey_pts.meandiffs[idx]
            p_val = tukey_pts.pvalues[idx]
            print(f"   • Temp. {season1:.0f} vs Temp. {season2:.0f}: "
                  f"diff={mean_diff:.3f}, p={p_val:.4f}")
            tukey_significant = True

if not tukey_significant:
    print("   No hay diferencias significativas entre temporadas específicas")

# ----------------------------------------------------
# 4. REGRESIÓN LINEAL PARA TENDENCIAS TEMPORALES
# ----------------------------------------------------
print("\n4. REGRESIÓN LINEAL: TENDENCIAS TEMPORALES")

import statsmodels.api as sm

# 4.1 Regresión para puntos
X_pts = sm.add_constant(df['SEASON_NUM'])
y_pts = df['PTS_LOG']
model_pts = sm.OLS(y_pts, X_pts).fit()

# 4.2 Regresión para tasa de triples
X_3pt = sm.add_constant(df['SEASON_NUM'])
y_3pt = df['FG3A_RATE']
model_3pt = sm.OLS(y_3pt, X_3pt).fit()

print("\n   REGRESIÓN - PUNTOS POR TEMPORADA:")
print(f"   • Ecuación: Puntos = {model_pts.params[0]:.3f} + {model_pts.params[1]:.3f}×Temporada")
print(f"   • R²: {model_pts.rsquared:.3f} ({model_pts.rsquared*100:.1f}% de varianza explicada)")
print(f"   • p-valor (pendiente): {model_pts.pvalues[1]:.6f}")
print(f"   • Cada temporada aumenta puntos en: {model_pts.params[1]:.3f} (log-scale)")

print("\n   REGRESIÓN - TASA DE TRIPLES:")
print(f"   • Ecuación: TripleRate = {model_3pt.params[0]:.3f} + {model_3pt.params[1]:.3f}×Temporada")
print(f"   • R²: {model_3pt.rsquared:.3f} ({model_3pt.rsquared*100:.1f}% de varianza explicada)")
print(f"   • p-valor (pendiente): {model_3pt.pvalues[1]:.6f}")
print(f"   • Cada temporada aumenta tasa de triples en: {model_3pt.params[1]:.4f}")

# ----------------------------------------------------
# 5. CORRELACIÓN ENTRE PUNTOS Y USO DE TRIPLES
# ----------------------------------------------------
print("\n5. CORRELACIÓN ENTRE PUNTOS Y TRIPLES")

corr_pts_3pt, p_corr_pts_3pt = stats.pearsonr(df['PTS_LOG'], df['FG3A_RATE'])
corr_season_pts, _ = stats.pearsonr(df['SEASON_NUM'], df['PTS_LOG'])
corr_season_3pt, _ = stats.pearsonr(df['SEASON_NUM'], df['FG3A_RATE'])

print(f"   • Correlación Puntos ↔ Tasa de Triples: {corr_pts_3pt:.3f} (p={p_corr_pts_3pt:.6f})")
print(f"   • Correlación Temporada ↔ Puntos: {corr_season_pts:.3f}")
print(f"   • Correlación Temporada ↔ Triples: {corr_season_3pt:.3f}")

# ----------------------------------------------------
# 6. VISUALIZACIONES
# ----------------------------------------------------
print("\n6. VISUALIZACIÓN DE TENDENCIAS TEMPORALES")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 6.1 Evolución de puntos
seasons = df_season['SEASON_NUM']
years = 2009 + seasons
axes[0, 0].plot(years, df_season['PTS_APPROX'], 'o-', linewidth=2, markersize=8, color='blue')
axes[0, 0].set_xlabel('Temporada NBA')
axes[0, 0].set_ylabel('Puntos por Partido (aprox.)')
axes[0, 0].set_title('Evolución de Puntos por Partido (2010-2024)')
axes[0, 0].grid(True, alpha=0.3)
axes[0, 0].tick_params(axis='x', rotation=45)

# Tendencia lineal
z = np.polyfit(seasons, df_season['PTS_APPROX'], 1)
p = np.poly1d(z)
axes[0, 0].plot(years, p(seasons), 'r--', alpha=0.7, 
                label=f'Tendencia: {z[0]:.1f} pts/año')

# 6.2 Evolución de tasa de triples
axes[0, 1].plot(years, df_season['FG3A_RATE']*100, 'o-', linewidth=2, markersize=8, color='green')
axes[0, 1].set_xlabel('Temporada NBA')
axes[0, 1].set_ylabel('Tasa de Triples (%)')
axes[0, 1].set_title('Evolución del Uso del Triple (2010-2024)')
axes[0, 1].grid(True, alpha=0.3)
axes[0, 1].tick_params(axis='x', rotation=45)

# Tendencia lineal
z2 = np.polyfit(seasons, df_season['FG3A_RATE']*100, 1)
p2 = np.poly1d(z2)
axes[0, 1].plot(years, p2(seasons), 'r--', alpha=0.7, 
                label=f'Tendencia: {z2[0]:.1f}%/año')

# 6.3 Boxplots por temporada (puntos)
season_data = [df[df['SEASON_NUM'] == s]['PTS_LOG'].values for s in sorted(df['SEASON_NUM'].unique())]
box = axes[1, 0].boxplot(season_data, patch_artist=True)
axes[1, 0].set_xlabel('Temporada (1=2010, 15=2024)')
axes[1, 0].set_ylabel('Puntos (log-transformados)')
axes[1, 0].set_title('Distribución de Puntos por Temporada')
axes[1, 0].grid(True, alpha=0.3, axis='y')

# Colorear boxes
colors = plt.cm.viridis(np.linspace(0, 1, len(season_data)))
for patch, color in zip(box['boxes'], colors):
    patch.set_facecolor(color)

# 6.4 Correlación puntos vs triples
scatter = axes[1, 1].scatter(df['PTS_LOG'], df['FG3A_RATE'], 
                            c=df['SEASON_NUM'], alpha=0.5, cmap='viridis')
axes[1, 1].set_xlabel('Puntos por Partido (log)')
axes[1, 1].set_ylabel('Tasa de Tiros de Triple')
axes[1, 1].set_title('Relación entre Puntos y Uso del Triple')
axes[1, 1].grid(True, alpha=0.3)

# Añadir línea de tendencia
z3 = np.polyfit(df['PTS_LOG'], df['FG3A_RATE'], 1)
p3 = np.poly1d(z3)
axes[1, 1].plot(sorted(df['PTS_LOG']), p3(sorted(df['PTS_LOG'])), 
                'r-', linewidth=2, alpha=0.7)

# Barra de color para temporadas
plt.colorbar(scatter, ax=axes[1, 1], label='Temporada')

plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 7. INTERPRETACIÓN EN CONTEXTO NBA
# ----------------------------------------------------
print("\n" + "="*80)
print("INTERPRETACIÓN DE RESULTADOS - REVOLUCIÓN OFENSIVA")
print("="*80)

# Calcular cambios absolutos
first_season = df_season.iloc[0]
last_season = df_season.iloc[-1]

pts_increase = last_season['PTS_APPROX'] - first_season['PTS_APPROX']
pts_pct_increase = (pts_increase / first_season['PTS_APPROX']) * 100

triples_increase = last_season['FG3A_RATE'] - first_season['FG3A_RATE']
triples_pct_increase = (triples_increase / first_season['FG3A_RATE']) * 100

print(f"\nCAMBIO ABSOLUTO 2010 → 2024:")
print("-"*50)
print(f"1. PUNTOS POR PARTIDO:")
print(f"   • 2010: {first_season['PTS_APPROX']:.1f} puntos")
print(f"   • 2024: {last_season['PTS_APPROX']:.1f} puntos")
print(f"   • Aumento: +{pts_increase:.1f} puntos ({pts_pct_increase:.1f}%)")

print(f"\n2. USO DEL TRIPLE:")
print(f"   • 2010: {first_season['FG3A_RATE']*100:.1f}% de tiros eran triples")
print(f"   • 2024: {last_season['FG3A_RATE']*100:.1f}% de tiros son triples")
print(f"   • Aumento: +{triples_increase*100:.1f} puntos porcentuales ({triples_pct_increase:.1f}%)")

print(f"\n3. TENDENCIAS ANUALES:")
print(f"   • Puntos aumentan {z[0]:.1f} por año en promedio")
print(f"   • Uso del triple aumenta {z2[0]:.2f}% por año")

# =============================================
# PREGUNTA 3: PREDICTORES DE PLUS_MINUS (REGRESIÓN MÚLTIPLE)
# =============================================

print("\n" + "="*80)
print("PREGUNTA 3: ¿QUÉ ESTADÍSTICAS PREDICEN EL DOMINIO EN UN PARTIDO?")
print("="*80)

# ----------------------------------------------------
# 1. SELECCIÓN DE VARIABLES PREDICTORAS
# ----------------------------------------------------
print("\n1. SELECCIÓN DE VARIABLES PREDICTORAS")

# Variables candidatas (estadísticas básicas estandarizadas)
predictors = ['AST_STD', 'REB_STD', 'STL_STD', 'BLK_STD', 'TOV_STD',
              'FG_PCT_MM', 'FG3A_RATE', 'OFF_EFF']

# Verificar disponibilidad
available_preds = [p for p in predictors if p in df.columns]
print(f"   • Predictores disponibles: {len(available_preds)} de {len(predictors)}")
for p in available_preds:
    print(f"     ✓ {p}")

# Variable respuesta: Diferencial de puntos transformado
y_plusminus = df['PLUS_MINUS_BOXCOX']

# ----------------------------------------------------
# 2. ANÁLISIS DE CORRELACIONES INICIAL
# ----------------------------------------------------
print("\n2. ANÁLISIS DE CORRELACIONES CON PLUS_MINUS")

correlations = []
for predictor in available_preds:
    corr, p_val = stats.pearsonr(df[predictor], y_plusminus)
    correlations.append({
        'Predictor': predictor,
        'Correlación': corr,
        'p-valor': p_val,
        'Significativo': p_val < 0.05
    })

corr_df = pd.DataFrame(correlations).sort_values('Correlación', ascending=False)

print("\n   CORRELACIONES INDIVIDUALES CON PLUS_MINUS:")
print("   " + "-"*60)
print(f"   {'Predictor':15s} {'Correlación':>12s} {'p-valor':>12s} {'Signif.':>10s}")
print("   " + "-"*60)

for _, row in corr_df.iterrows():
    sig = "✓" if row['Significativo'] else "✗"
    print(f"   {row['Predictor']:15s} {row['Correlación']:12.3f} {row['p-valor']:12.6f} {sig:>10s}")

print("   " + "-"*60)

# ----------------------------------------------------
# 3. REGRESIÓN LINEAL MÚLTIPLE COMPLETA
# ----------------------------------------------------
print("\n3. REGRESIÓN LINEAL MÚLTIPLE - MODELO COMPLETO")

import statsmodels.api as sm

# Preparar matriz de diseño
X_full = sm.add_constant(df[available_preds])
model_full = sm.OLS(y_plusminus, X_full).fit()

print(f"\n   MODELO COMPLETO (R² = {model_full.rsquared:.3f}):")
print("   " + "-"*80)
print(f"   {'Predictor':15s} {'Coef.':>10s} {'Error Est.':>12s} {'t':>8s} {'P>|t|':>10s} {'Importancia':>12s}")
print("   " + "-"*80)

# Calcular importancia relativa (coeficiente estandarizado)
coefs = model_full.params[1:]  # Excluir constante
std_devs = df[available_preds].std()
y_std = y_plusminus.std()
std_coefs = coefs * (std_devs / y_std)

for i, pred in enumerate(available_preds):
    coef = model_full.params[pred]
    std_err = model_full.bse[pred]
    t_val = model_full.tvalues[pred]
    p_val = model_full.pvalues[pred]
    importance = abs(std_coefs[pred]) if pred in std_coefs else 0
    
    sig = ""
    if p_val < 0.001: sig = "***"
    elif p_val < 0.01: sig = "**"
    elif p_val < 0.05: sig = "*"
    
    print(f"   {pred:15s} {coef:10.4f} {std_err:12.4f} {t_val:8.2f} {p_val:10.4f} {importance:12.3f} {sig}")

print("   " + "-"*80)
print(f"   Constante: {model_full.params['const']:.4f}")
print(f"   R²: {model_full.rsquared:.3f} | R² Ajustado: {model_full.rsquared_adj:.3f}")
print(f"   F-statistic: {model_full.fvalue:.2f} (p={model_full.f_pvalue:.6f})")
print("   *** p<0.001, ** p<0.01, * p<0.05")

# ----------------------------------------------------
# 4. SELECCIÓN DE VARIABLES PASO A PASO (STEPWISE)
# ----------------------------------------------------
print("\n4. SELECCIÓN DE VARIABLES - MÉTODO STEPWISE")

def forward_selection(X, y, significance_level=0.05):
    """Selección forward de variables"""
    initial_features = []
    selected_features = list(initial_features)
    
    current_score, best_new_score = 0.0, 0.0
    while available_preds and current_score == best_new_score:
        scores_with_candidates = []
        for candidate in available_preds:
            features = selected_features + [candidate]
            X_temp = sm.add_constant(X[features])
            model = sm.OLS(y, X_temp).fit()
            score = model.rsquared_adj
            scores_with_candidates.append((score, candidate))
        
        scores_with_candidates.sort()
        best_new_score, best_candidate = scores_with_candidates.pop()
        
        # Verificar si mejora significativamente
        if best_new_score > current_score:
            available_preds.remove(best_candidate)
            selected_features.append(best_candidate)
            current_score = best_new_score
    
    return selected_features, current_score

# Aplicar selección forward
selected_features, best_score = forward_selection(df[available_preds], y_plusminus)

print(f"\n   VARIABLES SELECCIONADAS (R² ajustado = {best_score:.3f}):")
for i, feature in enumerate(selected_features, 1):
    print(f"   {i:2d}. {feature}")

# ----------------------------------------------------
# 5. MODELO FINAL CON VARIABLES SELECCIONADAS
# ----------------------------------------------------
print("\n5. MODELO FINAL OPTIMIZADO")

# Modelo con variables seleccionadas
X_selected = sm.add_constant(df[selected_features])
model_final = sm.OLS(y_plusminus, X_selected).fit()

print(f"\n   MODELO FINAL (R² ajustado = {model_final.rsquared_adj:.3f}):")
print("   " + "-"*70)
for i, pred in enumerate(['const'] + selected_features):
    coef = model_final.params[pred]
    std_err = model_final.bse[pred] if pred != 'const' else 0
    t_val = model_final.tvalues[pred] if pred != 'const' else 0
    p_val = model_final.pvalues[pred] if pred != 'const' else 0
    
    if pred == 'const':
        print(f"   {pred:15s} {coef:10.4f}")
    else:
        sig = ""
        if p_val < 0.001: sig = "***"
        elif p_val < 0.01: sig = "**"
        elif p_val < 0.05: sig = "*"
        print(f"   {pred:15s} {coef:10.4f} {std_err:10.4f} {t_val:8.2f} {p_val:10.4f} {sig}")

print("   " + "-"*70)
print(f"   R²: {model_final.rsquared:.3f} | R² ajustado: {model_final.rsquared_adj:.3f}")
print(f"   F-statistic: {model_final.fvalue:.2f} (p={model_final.f_pvalue:.6f})")

# ----------------------------------------------------
# 6. VALIDACIÓN DEL MODELO - SUPUESTOS
# ----------------------------------------------------
print("\n6. VALIDACIÓN DE SUPUESTOS DEL MODELO")

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.outliers_influence import variance_inflation_factor

# 6.1 Homocedasticidad
bp_test = het_breuschpagan(model_final.resid, model_final.model.exog)
print(f"\n   HOMOCEDASTICIDAD (Breusch-Pagan):")
print(f"   • LM-statistic: {bp_test[0]:.3f}")
print(f"   • p-value: {bp_test[1]:.4f}")
print(f"   • Conclusión: {'Varianzas constantes (p>0.05)' if bp_test[1] > 0.05 else 'Problema de heterocedasticidad'}")

# 6.2 Multicolinealidad (VIF)
print(f"\n   MULTICOLINEALIDAD (VIF):")
vif_data = pd.DataFrame()
vif_data["Variable"] = selected_features
vif_data["VIF"] = [variance_inflation_factor(X_selected.values, i+1) 
                   for i in range(len(selected_features))]

for _, row in vif_data.iterrows():
    status = "OK" if row['VIF'] < 5 else "ALERTA" if row['VIF'] < 10 else "PROBLEMA"
    print(f"   • {row['Variable']:15s}: VIF = {row['VIF']:5.2f} ({status})")

# 6.3 Normalidad de residuos
shapiro_stat, shapiro_p = stats.shapiro(model_final.resid[:5000])  # Shapiro limitado a 5000
print(f"\n   NORMALIDAD DE RESIDUOS (Shapiro-Wilk):")
print(f"   • Estadístico: {shapiro_stat:.3f}")
print(f"   • p-value: {shapiro_p:.4f}")
print(f"   • Conclusión: {'Residuos normales (p>0.05)' if shapiro_p > 0.05 else 'Residuos no normales'}")

# ----------------------------------------------------
# 7. VISUALIZACIONES
# ----------------------------------------------------
print("\n7. VISUALIZACIÓN DE RESULTADOS")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 7.1 Importancia de predictores
importance_df = pd.DataFrame({
    'Predictor': selected_features,
    'Importancia': abs(std_coefs[selected_features].values)
}).sort_values('Importancia', ascending=True)

axes[0, 0].barh(importance_df['Predictor'], importance_df['Importancia'], color='steelblue')
axes[0, 0].set_xlabel('Importancia Relativa (coeficiente estandarizado)')
axes[0, 0].set_title('Importancia de Predictores en PLUS_MINUS')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# 7.2 Valores reales vs predichos
y_pred = model_final.predict(X_selected)
axes[0, 1].scatter(y_pred, y_plusminus, alpha=0.3, color='green')
axes[0, 1].plot([y_plusminus.min(), y_plusminus.max()], 
                [y_plusminus.min(), y_plusminus.max()], 
                'r--', linewidth=2, alpha=0.7)
axes[0, 1].set_xlabel('PLUS_MINUS Predicho')
axes[0, 1].set_ylabel('PLUS_MINUS Real')
axes[0, 1].set_title('Valores Reales vs Predichos (R² = {:.3f})'.format(model_final.rsquared))
axes[0, 1].grid(True, alpha=0.3)

# 7.3 Residuos vs valores predichos
axes[1, 0].scatter(y_pred, model_final.resid, alpha=0.3, color='purple')
axes[1, 0].axhline(y=0, color='red', linestyle='--', alpha=0.7)
axes[1, 0].set_xlabel('Valores Predichos')
axes[1, 0].set_ylabel('Residuos')
axes[1, 0].set_title('Análisis de Residuos (Homocedasticidad)')
axes[1, 0].grid(True, alpha=0.3)

# 7.4 QQ-plot de residuos
stats.probplot(model_final.resid, dist="norm", plot=axes[1, 1])
axes[1, 1].set_title('QQ-Plot - Normalidad de Residuos')
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ----------------------------------------------------
# 8. INTERPRETACIÓN EN CONTEXTO NBA
# ----------------------------------------------------
print("\n" + "="*80)
print("INTERPRETACIÓN DE RESULTADOS - PREDICTORES DE DOMINIO")
print("="*80)

# Convertir coeficientes a impacto práctico
print(f"\nIMPACTO PRÁCTICO EN EL RESULTADO:")
print("-"*50)

# Para cada predictor significativo (p < 0.05)
significant_preds = [(pred, model_final.params[pred], model_final.pvalues[pred]) 
                     for pred in selected_features 
                     if model_final.pvalues[pred] < 0.05]

significant_preds.sort(key=lambda x: abs(x[1]), reverse=True)

for pred, coef, p_val in significant_preds:
    # Calcular efecto de 1 desviación estándar
    std_effect = coef * df[pred].std()
    
    # Interpretar dirección
    direction = "AUMENTA" if coef > 0 else "DISMINUYE"
    effect_desc = f"{abs(std_effect):.2f} unidades de PLUS_MINUS"
    
    print(f"• {pred:15s}: {direction} el dominio del partido")
    print(f"  (1 desv. estándar en {pred} → {effect_desc})")

print("\n=== EJEMPLO PRÁCTICO - PARTIDO TÍPICO ===")
print("-" * 50)

# 1. OBTENER EL ORDEN EXACTO DE LAS VARIABLES DEL MODELO 
model_vars_ordered = [v for v in model_final.params.index if v != 'const']
print(f"Orden de variables del modelo: {model_vars_ordered}")

# 2. CREAR avg_values EN EL ORDEN CORRECTO
avg_values_ordered = {var: df[var].mean() for var in model_vars_ordered}

print("\nUn equipo con estadísticas PROMEDIO:")
for var, value in avg_values_ordered.items():
    print(f"  • {var:<12} : {value:.2f}")

# 3. CREAR DataFrame EN EL ORDEN CORRECTO
df_pred = pd.DataFrame([avg_values_ordered])[model_vars_ordered]  

# 4. VERIFICAR QUE LAS COLUMNAS COINCIDAN
print(f"\nVerificación de columnas:")
print(f"DataFrame columns: {df_pred.columns.tolist()}")
print(f"Modelo espera: {model_vars_ordered}")

# 5. AGREGAR CONSTANTE
df_pred_const = sm.add_constant(df_pred, has_constant='add')

# 6. VERIFICAR DIMENSIONES FINALES
print(f"\nDimensiones finales:")
print(f"df_pred_const shape: {df_pred_const.shape}")
print(f"Número de parámetros del modelo: {len(model_final.params)}")

# 7. REORDENAR SI ES NECESARIO (asegurar orden exacto)
if list(df_pred_const.columns) != list(model_final.params.index):
    print(" Reordenando columnas para coincidir con el modelo...")
    df_pred_const = df_pred_const[model_final.params.index]

# 8. REALIZAR PREDICCIÓN
try:
    avg_plusminus = model_final.predict(df_pred_const)[0]
    print(f" PREDICCIÓN EXITOSA")
    print(f"PLUS_MINUS predicho para equipo promedio: {avg_plusminus:.2f}")
    
    # Comparar con valor real
    if 'PLUS_MINUS' in df.columns:
        real_avg = df['PLUS_MINUS'].mean()
        print(f"PLUS_MINUS real promedio: {real_avg:.2f}")
        
except Exception as e:
    print(f"Error en predicción: {e}")
    
    # DEPURACIÓN DETALLADA
    print("\n=== DEPURACIÓN DETALLADA ===")
    print(f"Parámetros del modelo: {model_final.params.index.tolist()}")
    print(f"Columnas en df_pred_const: {df_pred_const.columns.tolist()}")
    
    # Verificar coincidencia
    for i, (param, col) in enumerate(zip(model_final.params.index, df_pred_const.columns)):
        match = "✓" if param == col else "✗"
        print(f"{match} {i}: Modelo={param}, DataFrame={col}")


