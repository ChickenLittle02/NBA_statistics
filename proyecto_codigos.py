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
# 16. TRANSFORMACIONES Y PREPROCESAMIENTO COMPLETO
# =============================================

print("\n" + "="*80)
print("TRANSFORMACIONES Y PREPROCESAMIENTO COMPLETO")
print("="*80)

# Primero verifica la discrepancia en el tamaño de datos
print(f"\n⚠️  DISCREPANCIA DETECTADA:")
print(f"   - Tu dataset tiene: {df.shape[0]:,} filas")
print(f"   - Los boxplots muestran: ~33,000 filas")
print("\n   Verifica que estás usando el archivo correcto: 'regular_season_totals_2010_2024.csv'")
print("   El dataset debería tener ~33,000 filas según los boxplots proporcionados.")

# Si el dataset es correcto, proceder con las transformaciones
if df.shape[0] > 1000:  # Si tiene suficientes filas
    # 1. IMPUTACIÓN DE VALORES FALTANTES
    print("\n1. IMPUTACIÓN DE VALORES FALTANTES")
    
    variables_imputar = ['PLUS_MINUS', 'FG_PCT', 'FT_PCT', 'FG3_PCT']
    for var in variables_imputar:
        if var in df.columns:
            n_missing = df[var].isnull().sum()
            if n_missing > 0:
                median_val = df[var].median()
                df[var].fillna(median_val, inplace=True)
                print(f"   ✓ {var}: {n_missing} valores imputados con mediana={median_val:.3f}")
    
    # 2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS
    print("\n2. CODIFICACIÓN DE VARIABLES CATEGÓRICAS")
    
    if 'WL' in df.columns:
        df['WL_BINARY'] = df['WL'].map({'W': 1, 'L': 0})
        print(f"   ✓ WL codificado a binario (1=Victoria, 0=Derrota)")
    
    if 'MATCHUP' in df.columns:
        # Extraer si es local (1) o visitante (0)
        df['IS_HOME'] = df['MATCHUP'].apply(lambda x: 0 if '@' in str(x) else 1)
        print(f"   ✓ Variable IS_HOME creada de MATCHUP")
    
    # 3. CREACIÓN DE VARIABLES DERIVADAS
    print("\n3. CREACIÓN DE VARIABLES DERIVADAS")
    
    # Verificar que existen las columnas necesarias
    if all(col in df.columns for col in ['FGA', 'OREB', 'TOV', 'FTA', 'MIN']):
        # Posesiones (fórmula simplificada)
        df['POSS'] = df['FGA'] - df['OREB'] + df['TOV'] + (0.4 * df['FTA'])
        
        # Ritmo de juego (PACE)
        df['PACE'] = df['POSS'] / df['MIN'] * 48
        
        # Eficiencia ofensiva (puntos por 100 posesiones)
        df['OFF_EFF'] = df['PTS'] / df['POSS'] * 100
        
        print(f"   ✓ Variables avanzadas creadas: POSS, PACE, OFF_EFF")
    
    if all(col in df.columns for col in ['AST', 'TOV']):
        df['AST_TO_RATIO'] = df['AST'] / df['TOV'].replace(0, np.nan)
        print(f"   ✓ Ratio AST/TOV creado")
    
    if all(col in df.columns for col in ['FG3A', 'FGA']):
        df['FG3A_RATE'] = df['FG3A'] / df['FGA']
        print(f"   ✓ Tasa de triples (FG3A_RATE) creada")
    
    # 4. TRANSFORMACIONES PARA NORMALIDAD
    print("\n4. TRANSFORMACIONES PARA MEJORAR NORMALIDAD")
    
    if 'PLUS_MINUS' in df.columns:
        # Box-Cox requiere valores positivos
        shift = abs(df['PLUS_MINUS'].min()) + 0.001
        from scipy.stats import boxcox
        df['PLUS_MINUS_BOXCOX'], _ = boxcox(df['PLUS_MINUS'] + shift)
        print(f"   ✓ PLUS_MINUS transformado con Box-Cox")
    
    # Transformación logarítmica para variables de conteo
    for var in ['PTS', 'AST', 'REB', 'FGM', 'FG3M']:
        if var in df.columns:
            df[f'{var}_LOG'] = np.log1p(df[var])  # log(1+x) para evitar log(0)
    
    print(f"   ✓ Transformación logarítmica aplicada a variables de conteo")
    
    # 5. ESTANDARIZACIÓN DE VARIABLES
    print("\n5. ESTANDARIZACIÓN DE VARIABLES")
    
    # Variables para estandarizar
    vars_to_scale = ['PTS', 'AST', 'REB', 'STL', 'BLK', 'TOV']
    vars_to_scale = [v for v in vars_to_scale if v in df.columns]
    
    if vars_to_scale:
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        scaled_array = scaler.fit_transform(df[vars_to_scale])
        scaled_df = pd.DataFrame(scaled_array, columns=[f'{v}_STD' for v in vars_to_scale])
        df = pd.concat([df, scaled_df], axis=1)
        print(f"   ✓ {len(vars_to_scale)} variables estandarizadas (Z-score)")
    
    # 6. NORMALIZACIÓN DE PORCENTAJES
    print("\n6. NORMALIZACIÓN DE PORCENTAJES")
    
    vars_to_norm = ['FG_PCT', 'FG3_PCT', 'FT_PCT']
    vars_to_norm = [v for v in vars_to_norm if v in df.columns]
    
    if vars_to_norm:
        from sklearn.preprocessing import MinMaxScaler
        minmax_scaler = MinMaxScaler()
        norm_array = minmax_scaler.fit_transform(df[vars_to_norm])
        norm_df = pd.DataFrame(norm_array, columns=[f'{v}_NORM' for v in vars_to_norm])
        df = pd.concat([df, norm_df], axis=1)
        print(f"   ✓ {len(vars_to_norm)} porcentajes normalizados [0,1]")
    
    # 7. TRATAMIENTO DE OUTLIERS (Winsorizing)
    print("\n7. TRATAMIENTO DE OUTLIERS")
    
    if 'PLUS_MINUS' in df.columns:
        from scipy.stats.mstats import winsorize
        df['PLUS_MINUS_WINS'] = winsorize(df['PLUS_MINUS'], limits=[0.05, 0.05])
        print(f"   ✓ PLUS_MINUS winsorizado al 5% y 95%")
    
    # 8. VERIFICACIÓN POST-TRANSFORMACIÓN
    print("\n8. VERIFICACIÓN POST-TRANSFORMACIÓN")
    print(f"   • Dimensiones finales: {df.shape}")
    print(f"   • Columnas totales: {len(df.columns)}")
    print(f"   • Nuevas columnas creadas: {len([c for c in df.columns if any(x in c for x in ['_STD', '_NORM', '_LOG', '_BOXCOX', '_WINS', 'BINARY', 'RATE', 'RATIO', 'EFF', 'PACE', 'POSS'])])}")
    
    # Exportar dataset transformado
    df.to_csv('nba_data_transformed.csv', index=False, encoding='utf-8-sig')
    print(f"\n✓ Dataset transformado exportado a 'nba_data_transformed.csv'")
    
    # Crear dataset solo con variables transformadas para modelado
    transformed_cols = [col for col in df.columns if any(x in col for x in ['_STD', '_NORM', '_LOG', '_BOXCOX', 'BINARY', 'RATE', 'RATIO', 'EFF', 'PACE', 'POSS', 'IS_HOME'])]
    df_model = df[transformed_cols]
    df_model.to_csv('nba_data_modeling.csv', index=False, encoding='utf-8-sig')
    print(f"✓ Dataset para modelado exportado a 'nba_data_modeling.csv' ({len(transformed_cols)} variables)")

else:
    print("\n❌ ERROR: El dataset parece demasiado pequeño.")
    print("   El dataset original debería tener ~33,000 filas.")
    print("   Verifica que estás usando el archivo correcto.")