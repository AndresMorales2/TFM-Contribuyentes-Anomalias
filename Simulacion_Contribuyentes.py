import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import StandardScaler
import random
from imblearn.over_sampling import SMOTE
# **************************************************************************************
# 0. Semilla para reproducibilidad
# **************************************************************************************
np.random.seed(42)
random.seed(42)

# **************************************************************************************
# 1. Dataset de Contribuyentes
# **************************************************************************************
n_grandes = 2536
ids_grandes = [f"Contribuyente_{i}" for i in range(1, n_grandes+1)]

ubicaciones = np.random.choice(
    ["Bogotá", "Antioquia", "Valle del Cauca", "Santander", "Cundinamarca", "Atlántico"],
    size=n_grandes
)
sectores = np.random.choice(
    ["Manufactura", "Comercio", "Servicios financieros", "Construcción", "Tecnología"],
    size=n_grandes
)
responsable_iva = np.random.choice([0,1], size=n_grandes, p=[0.3,0.7])
responsable_renta = np.random.choice([0,1], size=n_grandes, p=[0.1,0.9])

df_contribuyentes = pd.DataFrame({
    "ID": ids_grandes,
    "Ubicacion": ubicaciones,
    "Sector_CIIU": sectores,
    "Responsable_IVA": responsable_iva,
    "Responsable_Renta": responsable_renta
})

# **************************************************************************************
# 2. Dataset de Facturación Electrónica
# **************************************************************************************
ingresos = np.random.normal(loc=5e8, scale=2e8, size=n_grandes).clip(min=1e6)
iva_ventas = ingresos * 0.19
compras = np.random.normal(loc=3e8, scale=1.5e8, size=n_grandes).clip(min=5e5)
iva_compras = compras * 0.19

df_facturas = pd.DataFrame({
    "ID": ids_grandes,
    "Ingresos": ingresos.astype(int),
    "IVA_Ventas": iva_ventas.astype(int),
    "Compras": compras.astype(int),
    "IVA_Compras": iva_compras.astype(int)
})

# **************************************************************************************
# 3. Dataset Exógena (muestra reducida)
# **************************************************************************************
n_no_grandes = 1000
ids_no_grandes = [f"Contribuyente_NoGrande_{i}" for i in range(1, n_no_grandes+1)]

data_exogena = []
for contribuyente in ids_grandes[:300]:  
    n_reportes = np.random.randint(50, 100)
    for _ in range(n_reportes):
        if np.random.rand() < 0.5:
            id_informado = np.random.choice(ids_grandes)
            tipo = "Grande"
        else:
            id_informado = np.random.choice(ids_no_grandes)
            tipo = "No Grande"
        valor = np.random.normal(loc=2e7, scale=5e6)
        iva = valor * 0.19
        operacion = np.random.choice(["Compra", "Venta", "Servicio"])
        coincide = np.random.choice([0,1], p=[0.3,0.7])
        data_exogena.append([contribuyente, id_informado, tipo, operacion, int(valor), int(iva), coincide])

df_exogena = pd.DataFrame(data_exogena, columns=[
    "ID_Informante", "ID_Informado", "Tipo_Contribuyente_Informado",
    "Operacion", "Valor_Operacion", "IVA_Operacion", "Coincidencia_Factura"
])

# **************************************************************************************
# 4. Dataset Declaración de Renta
# **************************************************************************************
ingresos_renta = ingresos * np.random.uniform(0.8, 1.2, size=n_grandes)
costos_renta = compras * np.random.uniform(0.7, 1.3, size=n_grandes)
gastos_renta = np.random.normal(loc=1e8, scale=5e7, size=n_grandes).clip(min=1e6)
deducciones_renta = np.random.normal(loc=5e7, scale=2e7, size=n_grandes).clip(min=0)

coincide_fact = (np.abs(ingresos - ingresos_renta) < 1e7).astype(int)
coincide_exog = np.random.choice([0,1], size=n_grandes, p=[0.4,0.6])

df_renta = pd.DataFrame({
    "ID": ids_grandes,
    "Ingresos_Reportados": ingresos_renta.astype(int),
    "Costos_Reportados": costos_renta.astype(int),
    "Gastos_Reportados": gastos_renta.astype(int),
    "Deducciones_Reportadas": deducciones_renta.astype(int),
    "Coincidencia_Facturacion": coincide_fact,
    "Coincidencia_Exogena": coincide_exog
})

# **************************************************************************************
# 5. Exploración visual (gráficos)
# **************************************************************************************

# Histograma de ingresos
plt.figure(figsize=(12,5))
plt.hist(df_facturas['Ingresos'], bins=30, color='skyblue', edgecolor='black')
plt.title("Distribución de Ingresos (Facturación)")
plt.xlabel("Ingresos")
plt.ylabel("Frecuencia")
plt.show()

# Histograma de compras
plt.figure(figsize=(12,5))
plt.hist(df_facturas['Compras'], bins=30, color='salmon', edgecolor='black')
plt.title("Distribución de Compras (Facturación)")
plt.xlabel("Compras")
plt.ylabel("Frecuencia")
plt.show()

# Boxplot de ingresos por sector
df_facturas_sector = df_facturas.merge(df_contribuyentes[['ID','Sector_CIIU']], on='ID')
plt.figure(figsize=(12,6))
df_facturas_sector.boxplot(column='Ingresos', by='Sector_CIIU', rot=45)
plt.title("Ingresos por Sector CIIU")
plt.suptitle("")
plt.xlabel("Sector")
plt.ylabel("Ingresos")
plt.show()

# Scatter plot ingresos vs costos
plt.figure(figsize=(8,6))
plt.scatter(df_renta['Ingresos_Reportados'], df_renta['Costos_Reportados'], alpha=0.5)
plt.title("Ingresos vs Costos (Declaración de Renta)")
plt.xlabel("Ingresos Reportados")
plt.ylabel("Costos Reportados")
plt.show()

# Heatmap de correlaciones
variables = df_renta[['Ingresos_Reportados','Costos_Reportados','Gastos_Reportados','Deducciones_Reportadas']]
plt.figure(figsize=(8,6))
sns.heatmap(variables.corr(), annot=True, cmap="coolwarm")
plt.title("Matriz de Correlación - Declaración de Renta")
plt.show()

# **************************************************************************************
# 6. Detección de anomalías
# **************************************************************************************
df_model = pd.DataFrame({
    "Ingresos_Facturacion": df_facturas["Ingresos"],
    "Compras_Facturacion": df_facturas["Compras"],
    "Ingresos_Renta": df_renta["Ingresos_Reportados"],
    "Costos_Renta": df_renta["Costos_Reportados"],
    "Gastos_Renta": df_renta["Gastos_Reportados"],
    "Deducciones_Renta": df_renta["Deducciones_Reportadas"]
})

scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_model)

# Isolation Forest
iso = IsolationForest(contamination=0.1, random_state=42)
df_renta["Anomalia_IF"] = iso.fit_predict(X_scaled)

# Local Outlier Factor
lof = LocalOutlierFactor(n_neighbors=20, contamination=0.1)
df_renta["Anomalia_LOF"] = lof.fit_predict(X_scaled)

# Resultados
print("Resultados de detección de anomalías:")
print(df_renta[["ID","Anomalia_IF","Anomalia_LOF"]].head(20))

anom_if = (df_renta["Anomalia_IF"]==-1).sum()
anom_lof = (df_renta["Anomalia_LOF"]==-1).sum()
print(f"Contribuyentes detectados como anomalías por Isolation Forest: {anom_if}")
print(f"Contribuyentes detectados como anomalías por LOF: {anom_lof}")

# Conteo de normales y anómalos
labels = ['Normales','Anómalos']
values = [(df_renta["Anomalia_IF"]==1).sum(), (df_renta["Anomalia_IF"]==-1).sum()]
colors = ['skyblue','salmon']

# Gráfico de torta
plt.figure(figsize=(6,6))
plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title("Distribución de contribuyentes según Isolation Forest")
plt.show()

# Comparación de Algoritmos
coinciden = ((df_renta["Anomalia_IF"]==-1) & (df_renta["Anomalia_LOF"]==-1)).sum()
solo_if = ((df_renta["Anomalia_IF"]==-1) & (df_renta["Anomalia_LOF"]==1)).sum()
solo_lof = ((df_renta["Anomalia_IF"]==1) & (df_renta["Anomalia_LOF"]==-1)).sum()

labels = ['Ambos','Solo IF','Solo LOF']
values = [coinciden, solo_if, solo_lof]

plt.bar(labels, values, color=['purple','blue','green'])
plt.title("Comparación de anomalías detectadas por IF y LOF")
plt.ylabel("Número de contribuyentes")
plt.show()

#Anomalías resaltadas
plt.figure(figsize=(8,6))
plt.scatter(df_renta['Ingresos_Reportados'], df_renta['Costos_Reportados'], 
            c=(df_renta['Anomalia_IF']==-1), cmap='coolwarm', alpha=0.5)
plt.title("Ingresos vs Costos con anomalías (Isolation Forest)")
plt.xlabel("Ingresos Reportados")
plt.ylabel("Costos Reportados")
plt.show()

# Distribución secctorial de anomalías
df_sector = df_contribuyentes.merge(df_renta[['ID','Anomalia_IF']], on='ID')
anom_por_sector = df_sector[df_sector['Anomalia_IF']==-1]['Sector_CIIU'].value_counts()

anom_por_sector.plot(kind='bar', color='salmon')
plt.title("Anomalías detectadas por sector (Isolation Forest)")
plt.ylabel("Número de anomalías")
plt.show()

# **************************************************************************************
# 7. Balanceo de datos con SMOTE
# **************************************************************************************

# Creamos etiquetas binarias a partir de Isolation Forest
y_labels = (df_renta["Anomalia_IF"]==-1).astype(int)  # 1 = anomalía, 0 = normal

# Aplicamos SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_scaled, y_labels)

print("Tamaño original:", X_scaled.shape)
print("Tamaño balanceado:", X_resampled.shape)

# Visualización del balanceo
labels = ['Normales','Anómalos']
values = [sum(y_resampled==0), sum(y_resampled==1)]
colors = ['skyblue','salmon']

plt.figure(figsize=(6,6))
plt.pie(values, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
plt.title("Distribución balanceada de contribuyentes (SMOTE)")
plt.show()

# **************************************************************************************
# 8. Validación de resultados (Excel enriquecido)
# **************************************************************************************

# Filtrar contribuyentes detectados como anomalías por ambos algoritmos
df_validacion = df_renta[(df_renta["Anomalia_IF"]==-1) & (df_renta["Anomalia_LOF"]==-1)].copy()

# Crear columna con motivo de riesgo
df_validacion["Motivo_Riesgo"] = "Detectado por IF y LOF"

# Definir fuente de riesgo según coincidencias
def fuente_riesgo(row):
    if row["Coincidencia_Facturacion"] == 0 and row["Coincidencia_Exogena"] == 0:
        return "Discrepancia en Facturación y Exógena"
    elif row["Coincidencia_Facturacion"] == 0:
        return "Discrepancia en Facturación"
    elif row["Coincidencia_Exogena"] == 0:
        return "Discrepancia en Exógena"
    else:
        return "Inconsistencias en Renta"

df_validacion["Fuente_Riesgo"] = df_validacion.apply(fuente_riesgo, axis=1)

# Tomar una muestra de 20 contribuyentes
muestra_ids = df_validacion["ID"].sample(20, random_state=42)

# Extraer información de todos los datasets
muestra_contribuyentes = df_contribuyentes[df_contribuyentes["ID"].isin(muestra_ids)]
muestra_facturas = df_facturas[df_facturas["ID"].isin(muestra_ids)]
muestra_renta = df_renta[df_renta["ID"].isin(muestra_ids)]

# Unir la información en un solo DataFrame
df_muestra = muestra_contribuyentes.merge(muestra_facturas, on="ID") \
                                   .merge(muestra_renta, on="ID") \
                                   .merge(df_validacion[["ID","Motivo_Riesgo","Fuente_Riesgo"]], on="ID")

# Exportar a Excel
df_muestra.to_excel("Muestra_Contribuyentes_Alto_Riesgo.xlsx", index=False)

print("Archivo 'Muestra_Contribuyentes_Alto_Riesgo.xlsx' generado con éxito.")

# **************************************************************************************
# 9. Ajuste de hiperparámetros
# **************************************************************************************

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor

# Definir los hiperparámetros a probar
contamination_values = [0.05, 0.1, 0.15]   # proporción de anomalías esperadas
n_neighbors_values = [20, 35, 50]          # número de vecinos para LOF

resultados_if = {}
resultados_lof = {}

# Ajuste de Isolation Forest
for c in contamination_values:
    modelo_if = IsolationForest(contamination=c, random_state=42)
    df_renta[f"Anomalia_IF_c{c}"] = modelo_if.fit_predict(df_renta[["Ingresos_Reportados",
                                                                    "Costos_Reportados",
                                                                    "Gastos_Reportados",
                                                                    "Deducciones_Reportadas"]])
    resultados_if[c] = (df_renta[f"Anomalia_IF_c{c}"]==-1).sum()

# Ajuste de Local Outlier Factor
for n in n_neighbors_values:
    modelo_lof = LocalOutlierFactor(n_neighbors=n)
    df_renta[f"Anomalia_LOF_n{n}"] = modelo_lof.fit_predict(df_renta[["Ingresos_Reportados",
                                                                      "Costos_Reportados",
                                                                      "Gastos_Reportados",
                                                                      "Deducciones_Reportadas"]])
    resultados_lof[n] = (df_renta[f"Anomalia_LOF_n{n}"]==-1).sum()

# Mostrar resultados comparativos
print("Resultados de Isolation Forest con distintos contamination:")
for c, count in resultados_if.items():
    print(f"Contamination={c}: {count} anomalías detectadas")

print("\nResultados de Local Outlier Factor con distintos n_neighbors:")
for n, count in resultados_lof.items():
    print(f"n_neighbors={n}: {count} anomalías detectadas")

# Validación frente a la muestra de 20 contribuyentes
muestra_ids = df_validacion["ID"].sample(20, random_state=42)
df_muestra_validacion = df_renta[df_renta["ID"].isin(muestra_ids)]

print("\nValidación de la muestra de 20 contribuyentes con hiperparámetros ajustados:")
print(df_muestra_validacion[["ID",
                             "Anomalia_IF_c0.05",
                             "Anomalia_IF_c0.1",
                             "Anomalia_IF_c0.15",
                             "Anomalia_LOF_n20",
                             "Anomalia_LOF_n35",
                             "Anomalia_LOF_n50"]].head(20))


# **************************************************************************************
# 10 Evaluación final (Excel + Gráfica con muestra de 20)
# **************************************************************************************

import matplotlib.pyplot as plt

# Unir datasets para tener todas las columnas originales
df_final = df_contribuyentes.merge(df_facturas, on="ID") \
                            .merge(df_renta, on="ID")

# Añadir resultados de hiperparámetros
df_final["Anomalia_IF_c0.05"] = df_renta["Anomalia_IF_c0.05"]
df_final["Anomalia_IF_c0.1"]  = df_renta["Anomalia_IF_c0.1"]
df_final["Anomalia_IF_c0.15"] = df_renta["Anomalia_IF_c0.15"]
df_final["Anomalia_LOF_n20"]  = df_renta["Anomalia_LOF_n20"]
df_final["Anomalia_LOF_n35"]  = df_renta["Anomalia_LOF_n35"]
df_final["Anomalia_LOF_n50"]  = df_renta["Anomalia_LOF_n50"]

# Añadir columnas de motivo y fuente de riesgo
df_final["Motivo_Riesgo"] = df_validacion["Motivo_Riesgo"]
df_final["Fuente_Riesgo"] = df_validacion["Fuente_Riesgo"]

# Seleccionar muestra de 20 contribuyentes
muestra_final = df_final.sample(20, random_state=42)

# Seleccionar columnas relevantes para exportar
columnas_exportar = ["ID","Ubicacion","Sector_CIIU","Responsable_IVA","Responsable_Renta",
                     "Ingresos","IVA_Ventas","Compras","IVA_Compras",
                     "Ingresos_Reportados","Costos_Reportados","Gastos_Reportados","Deducciones_Reportadas",
                     "Coincidencia_Facturacion","Coincidencia_Exogena",
                     "Anomalia_IF_c0.05","Anomalia_IF_c0.1","Anomalia_IF_c0.15",
                     "Anomalia_LOF_n20","Anomalia_LOF_n35","Anomalia_LOF_n50",
                     "Motivo_Riesgo","Fuente_Riesgo"]

# Exportar a Excel
muestra_final[columnas_exportar].to_excel("Evaluacion_Final_Muestra20.xlsx", index=False)
print("Archivo 'Evaluacion_Final_Muestra20.xlsx' generado con éxito.")

# **************************************************************************************
# Gráfica comparativa de anomalías
# **************************************************************************************

conteos_if = {
    "IF_c0.05": (df_renta["Anomalia_IF_c0.05"]==-1).sum(),
    "IF_c0.1": (df_renta["Anomalia_IF_c0.1"]==-1).sum(),
    "IF_c0.15": (df_renta["Anomalia_IF_c0.15"]==-1).sum()
}

conteos_lof = {
    "LOF_n20": (df_renta["Anomalia_LOF_n20"]==-1).sum(),
    "LOF_n35": (df_renta["Anomalia_LOF_n35"]==-1).sum(),
    "LOF_n50": (df_renta["Anomalia_LOF_n50"]==-1).sum()
}

plt.figure(figsize=(10,6))
plt.bar(conteos_if.keys(), conteos_if.values(), color="darkred", label="Isolation Forest")
plt.bar(conteos_lof.keys(), conteos_lof.values(), color="navy", label="Local Outlier Factor")

plt.title("Comparación de anomalías detectadas (Evaluación Final)")
plt.ylabel("Número de contribuyentes detectados como anomalías")
plt.xlabel("Configuración de hiperparámetros")
plt.legend()
plt.show()

# **************************************************************************************
# 10.1 Comparación con los mismos 20 iniciales
# **************************************************************************************

# Usar los mismos IDs de la muestra inicial (punto 5.8)
muestra_ids_inicial = df_validacion["ID"].sample(20, random_state=42)  # mismos que se usaron antes

# Filtrar esos contribuyentes en el DataFrame final
muestra_comparativa = df_final[df_final["ID"].isin(muestra_ids_inicial)]

# Seleccionar columnas relevantes para comparar antes vs después
columnas_comparar = ["ID","Ubicacion","Sector_CIIU",
                     "Coincidencia_Facturacion","Coincidencia_Exogena",
                     "Anomalia_IF","Anomalia_LOF",   # resultados iniciales
                     "Anomalia_IF_c0.05","Anomalia_IF_c0.1","Anomalia_IF_c0.15",
                     "Anomalia_LOF_n20","Anomalia_LOF_n35","Anomalia_LOF_n50",
                     "Motivo_Riesgo","Fuente_Riesgo"]

# Exportar tabla comparativa a Excel
muestra_comparativa[columnas_comparar].to_excel("Comparacion_Muestra20_Inicial_vs_Final.xlsx", index=False)

print("Archivo 'Comparacion_Muestra20_Inicial_vs_Final.xlsx' generado con éxito.")