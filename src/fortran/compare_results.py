#!/usr/bin/env python3
"""
Script para comparar los resultados de la versión Python y Fortran
del procesador de radar HF.

Uso:
    python compare_results.py archivo_python.txt archivo_fortran.txt
"""

import sys
import numpy as np
import pandas as pd

def read_results(filename):
    """Lee un archivo de resultados, ignorando líneas de comentario."""
    data = []
    with open(filename, 'r') as f:
        for line in f:
            if not line.startswith('#'):
                parts = line.strip().split()
                if len(parts) >= 7:  # Esperamos al menos 7 columnas
                    try:
                        data.append([float(x) for x in parts[:7]])
                    except ValueError:
                        continue  # Saltar cabecera
    
    return np.array(data)

def compare_results(file1, file2, tolerance=1e-5):
    """Compara dos archivos de resultados."""
    
    print("="*70)
    print("Comparación de Resultados: Python vs Fortran")
    print("="*70)
    
    # Leer archivos
    print(f"\nLeyendo {file1}...")
    data1 = read_results(file1)
    
    print(f"Leyendo {file2}...")
    data2 = read_results(file2)
    
    # Verificar dimensiones
    print(f"\nPuntos en archivo 1: {data1.shape[0]}")
    print(f"Puntos en archivo 2: {data2.shape[0]}")
    
    if data1.shape[0] != data2.shape[0]:
        print("\n⚠️  ADVERTENCIA: Número diferente de puntos!")
        n_points = min(data1.shape[0], data2.shape[0])
        print(f"Comparando solo los primeros {n_points} puntos comunes")
        data1 = data1[:n_points]
        data2 = data2[:n_points]
    
    # Nombres de columnas
    columns = ['longitude', 'latitude', 'u_total', 'v_total', 'modulo', 'angulo', 'gdop']
    
    print("\n" + "="*70)
    print("Diferencias por columna:")
    print("="*70)
    
    all_close = True
    
    for i, col in enumerate(columns):
        diff = np.abs(data1[:, i] - data2[:, i])
        max_diff = np.max(diff)
        mean_diff = np.mean(diff)
        
        # Diferencia relativa
        rel_diff = diff / (np.abs(data1[:, i]) + 1e-10)
        max_rel_diff = np.max(rel_diff) * 100  # En porcentaje
        
        status = "✓ OK" if max_diff < tolerance else "✗ DIFERENCIA"
        
        print(f"\n{col:15s}")
        print(f"  Diferencia máxima:     {max_diff:12.6e}  {status}")
        print(f"  Diferencia promedio:   {mean_diff:12.6e}")
        print(f"  Diferencia rel. máx:   {max_rel_diff:12.6f}%")
        
        if max_diff >= tolerance:
            all_close = False
            # Encontrar el punto con mayor diferencia
            idx = np.argmax(diff)
            print(f"  Punto con mayor diff:  índice {idx}")
            print(f"    Archivo 1: {data1[idx, i]:12.6f}")
            print(f"    Archivo 2: {data2[idx, i]:12.6f}")
    
    print("\n" + "="*70)
    if all_close:
        print("✓ RESULTADOS IDÉNTICOS (dentro de la tolerancia)")
    else:
        print(f"✗ DIFERENCIAS ENCONTRADAS (tolerancia: {tolerance})")
    print("="*70)
    
    # Estadísticas globales
    print("\nEstadísticas Globales:")
    print("-"*70)
    
    for i, col in enumerate(['u_total', 'v_total', 'modulo']):
        idx = columns.index(col)
        print(f"\n{col}:")
        print(f"  Archivo 1 - Media: {np.mean(data1[:, idx]):10.3f}  Std: {np.std(data1[:, idx]):10.3f}")
        print(f"  Archivo 2 - Media: {np.mean(data2[:, idx]):10.3f}  Std: {np.std(data2[:, idx]):10.3f}")
    
    return all_close

if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Uso: python compare_results.py archivo_python.txt archivo_fortran.txt")
        sys.exit(1)
    
    file1 = sys.argv[1]
    file2 = sys.argv[2]
    
    try:
        match = compare_results(file1, file2, tolerance=1e-4)
        sys.exit(0 if match else 1)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(2)
