import numpy as np
import pandas as pd

def visualize_table(table, m, n, basic_vars):
    # Actualizar los nombres de las filas según las variables básicas actuales.
    row_names = []
    for i in range(m):
        var_index = basic_vars[i]
        if var_index < (n - m):
            name = f"x{var_index+1}"
        else:
            name = f"s{var_index - (n - m) + 1}"
        row_names.append(name)
    row_names.append("z")
    
    # Las columnas permanecen fijas.
    col_names = [f"x{j+1}" for j in range(n)] + ["RHS"]
    df = pd.DataFrame(table, index=row_names, columns=col_names)
    print(df)

def simplex(table, m, n):
    basic_vars = list(range(n - m, n))
    iteration = 0
    while True:
        iteration += 1
        print(f"\nIteración: {iteration}")
        visualize_table(table, m, n, basic_vars)
        
        # Extraer la fila objetivo (última fila, sin incluir RHS)
        objective_row = table[m, :n]
        if all(objective_row >= 0):
            break

        # Seleccionar la variable entrante (columna con el coeficiente más negativo)
        pivot_col = np.argmin(objective_row)

        # Calcular ratios para determinar la fila pivote
        ratios = []
        for i in range(m):
            if table[i, pivot_col] > 0:
                ratios.append(table[i, n] / table[i, pivot_col])
            else:
                ratios.append(np.inf)
        ratios = np.array(ratios)
        pivot_row = np.argmin(ratios)

        # Realizar la operación de pivoteo: normalizar la fila pivote
        pivot_element = table[pivot_row, pivot_col]
        table[pivot_row, :] /= pivot_element

        # Actualizar todas las demás filas para anular los elementos en la columna pivote
        for i in range(m + 1):
            if i != pivot_row:
                table[i, :] -= table[i, pivot_col] * table[pivot_row, :]

        # Actualizar la base: la variable correspondiente a pivot_col entra en la base
        basic_vars[pivot_row] = pivot_col

    return table, basic_vars
