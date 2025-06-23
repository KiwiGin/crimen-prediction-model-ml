import pandas as pd

# Carga el dataset
df = pd.read_csv("AxonCrimeData_Export_WA_1686331975127960619 (1).csv")  # cambia esto por tu ruta

# Asegúrate de poner el nombre correcto de la columna que contiene el tipo de crimen
# Por ejemplo: 'crime_type', 'offense', 'offense_type', etc.
print("Columnas disponibles:", df.columns)

unique_crimes = df['NIBRS_Offense'].unique()

# Mostrar todos los tipos de crímenes
print("Tipos de crímenes en el dataset:")
for idx, crime in enumerate(unique_crimes):
    print(f"{idx}: {crime}")
