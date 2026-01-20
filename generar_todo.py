# ===========================================
# GENERAR TODO - Script Maestro
# ===========================================
# Ejecuta todos los análisis y genera los archivos .tex

import subprocess
import sys
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

scripts = [
    'pregunta1_logistica.py',
    'pregunta2_analisis.py', 
    'pregunta3_multiple.py'
]

print("=" * 50)
print("GENERANDO ANÁLISIS ESTADÍSTICOS")
print("=" * 50)

for script in scripts:
    print(f"\n>> Ejecutando {script}...")
    result = subprocess.run([sys.executable, script], capture_output=True, text=True)
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"ERROR en {script}:")
        print(result.stderr)

print("\n" + "=" * 50)
print("ARCHIVOS GENERADOS EN generated/:")
print("=" * 50)
for f in os.listdir('generated'):
    print(f"  [OK] {f}")

print("\n>> Ahora ejecuta: pdflatex proyecto.tex")
