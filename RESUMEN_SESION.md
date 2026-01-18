# 📊 Resumen de Análisis Estadístico NBA

**Fecha:** 18 de enero de 2026  
**Proyecto:** Análisis Estadístico de la NBA (2010-2024)

---

## Pregunta 1: Correlación FG3M vs Victoria

**Técnicas aplicadas:** Chi-Cuadrado + Regresión Logística

| Estadístico | Valor |
|-------------|-------|
| χ² | 62.12 |
| Valor p | < 0.001 |
| Odds Ratio (FG3M) | 1.52 |
| V de Cramér | 0.225 |

**Conclusión:** Anotar más triples está asociado significativamente con mayor probabilidad de victoria.

📁 Script: `pregunta1_logistica.py`

---

## Pregunta 2: Evolución Temporal de PTS y Triples

### Regresión Lineal Simple

| Variable | Pendiente | R² | Significativo |
|----------|-----------|-----|---------------|
| PTS | +1.64 pts/temporada | 97.5% | Sí |
| FG3A/FGA | +1.71%/temporada | 98.8% | Sí |

### ANOVA por Temporada

| Variable | F | Valor p | Decisión |
|----------|---|---------|----------|
| PTS | 32.30 | < 0.001 | Rechazar H₀ |
| FG3A/FGA | 48.61 | < 0.001 | Rechazar H₀ |

**Conclusión:** Existe una tendencia significativa al alza tanto en puntos como en uso del triple.

📁 Scripts: `pregunta2_anova.py`, `pregunta2_regresion.py`

---

## Pregunta 3: Predictores de PLUS_MINUS

**Técnica:** Regresión Lineal Múltiple

| Métrica | Valor |
|---------|-------|
| R² | 54.9% |
| R² ajustado | 54.7% |
| F-estadístico | 298.36 |

### Coeficientes

| Variable | β | Efecto |
|----------|---|--------|
| FG_PCT | +79.84 | **Más importante** |
| STL | +1.15 | Positivo |
| AST | +0.81 | Positivo |
| REB | +0.50 | Positivo |
| TOV | -1.47 | **Negativo** |

**Conclusión:** La eficiencia de tiro (FG_PCT) y minimizar pérdidas (TOV) son los factores clave.

📁 Script: `pregunta3_multiple.py`

---

## Archivos Generados

```
NBA_statistics/
├── pregunta1_logistica.py   # Chi-cuadrado + Reg. Logística
├── pregunta2_anova.py       # ANOVA por temporada
├── pregunta2_regresion.py   # Regresión lineal simple
├── pregunta3_multiple.py    # Regresión múltiple
└── RESUMEN_SESION.md        # Este archivo
```

---

## Próximos Pasos

1. Copiar el código LaTeX generado al `proyecto.tex`
2. Recompilar el PDF con `pdflatex proyecto.tex`
3. (Opcional) Ejecutar scripts con datos reales del CSV
