import pandas as pd
import sqlite3

# Cargar el CSV con el separador correcto
df = pd.read_csv("Hurtos_Personas_Medellin.csv", sep=";")

# Conectar o crear la base de datos SQLite
conn = sqlite3.connect("hurtos.db")

# Guardar el DataFrame como tabla en SQLite
df.to_sql("hurtos", conn, if_exists="replace", index=False)

# Cerrar conexión
conn.close()
