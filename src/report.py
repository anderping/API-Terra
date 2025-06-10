import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os
import uuid
from datetime import datetime, timedelta

from pymongo import MongoClient, DESCENDING


mi_paleta = [
    
    "#189B5C",  # Emerald Green
    #"#7CE55E",  # Neon Green
    "#FFB41D",  # Sol Yellow
    "#3D9DD8",  # Blue Cielo
    "#F96E43",  # Rojizo
    "#F78BD8",  # Blushed
    "#FFE9BB",  # Sol Yellow 70%
    "#C5E2F3",  # Blue Cielo 70%
    "#FDD4C7",  # Rojizo 70%
    "#FDDCF3",  # Blushed 70%
]


def save_and_close_fig(fig, output_dir, filenames):
    os.makedirs(output_dir, exist_ok=True)

    filename = f"graph_{uuid.uuid4().hex}.png"
    path = os.path.join(output_dir, filename)

    fig.savefig(path, transparent=True)
    plt.close(fig)

    filenames.append(filename)


def generate_report(data, frequency='weekly'):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, 'static', 'reports')
    os.makedirs(output_dir, exist_ok=True)
    
    # Elimina imágenes previas del directorio
    for fname in os.listdir(output_dir):
        if fname.endswith(".png"):
            os.remove(os.path.join(output_dir, fname))

    # # Obtener datos de MongoDB
    # client = MongoClient("mongodb://localhost:27017/") # CAMBIAR SEGÚN URL DE CONEXIÓN
    # db = client["mi_base"] # CAMBIAR SEGÚN NOMBRE DE BASE DE DATOS
    # coleccion = db["mi_coleccion"]  # CAMBIAR SEGÚN NOMBRE DE COLECCIÓN

    # projection = {
    #     "issueType": 1,
    #     "status": 1,
    #     "client.name": 1,
    # }

    # if frequency == 'weekly':
    #     # Informe semanal regular: últimos 30 registros
    #     datos = list(coleccion.find({}, projection).sort("createdAt", DESCENDING).limit(30))  # HAY QUE ASEGURAR CON FULLSTACK QUE "DATE" ESTÁ EN FORMATO CORRECTO (DATE)

    # elif frequency == 'monthly':
    #     # Informe mensual regular: últimos 120 registros
    #     datos = list(coleccion.find({}, projection).sort("createdAt", DESCENDING).limit(120))

    # if not datos:
    #     raise ValueError("No data available.")
    
    # for d in datos:
    #     d["client_name"] = d.get("client", {}).get("name")
    #     d.pop("client", None)  # Elimina el objeto client si ya no se necesita

    df = pd.DataFrame(data) # ASEGURAR QUE ESE ES EL NOMBRE DEL DATAFRAME


    filenames = []
    
    # GENERAR GRÁFICOS

    # GRÁFICO 1: Distribución de tipos de solicitudes
    
    # Convertir 'created_at' a datetime
    df['createdAt'] = pd.to_datetime(df['createdAt'])

    # Calcular la fecha de hace una semana
    hoy = datetime.now()
    una_semana_atras = hoy - timedelta(days=7)

    # Filtrar las solicitudes de la última semana
    df_semana = df[df['createdAt'] >= una_semana_atras]

    # Contar incidencias por tipo
    conteo_por_tipo = df_semana['issueType'].value_counts()
    total = conteo_por_tipo.sum()

    # Calcular porcentajes
    porcentajes = (conteo_por_tipo / total * 100).round(2)

    # Explode para destacar cada porción
    explode = [0.05] * len(porcentajes)

    # Crear gráfico de pastel
    fig, ax = plt.subplots(figsize=(8, 6))
    wedges, texts, autotexts = ax.pie(
        porcentajes,
        labels=None,  # No mostramos etiquetas directamente
        autopct='%1.1f%%',
        explode=explode,
        shadow=True,
        startangle=140,
        textprops={'fontsize': 10, 'weight': 'bold'}
    )

    # Crear leyenda
    ax.legend(
        wedges,
        conteo_por_tipo.index,
        title="Tipo de Solicitud",
        loc="center left",
        bbox_to_anchor=(1, 0, 0.5, 1),
        frameon=False,
    )

    #ax.set_title("Solicitudes recibidas en la última semana", fontsize=14, weight='bold')
    ax.set_title(f"Distribución de Solicitudes\nSemana previa al {hoy.strftime('%d/%m/%Y')}", pad=20)
    ax.axis('equal')  # Para que sea un círculo

    # Eliminar fondo
    fig.patch.set_visible(False)
    ax.set_facecolor('none')

    plt.tight_layout()
    plt.show()

    # Guardar gráfico
    save_and_close_fig(fig, output_dir, filenames)



    # GRÁFICO 2: Cantidad de solicitudes por estado

    hoy = datetime.now()
    una_semana_atras = hoy - timedelta(days=10)
    df_ultima_semana = df[df['created_at'] >= una_semana_atras]

    df_muestra = df_ultima_semana.sample(n=min(30, len(df_ultima_semana)), random_state=42)

    conteo_por_estado = df_muestra['status'].value_counts()

    fig, ax = plt.subplots(figsize=(8, 5))  # Usa subplots para tener control sobre el eje
    ax.bar(conteo_por_estado.index, conteo_por_estado.values, color=mi_paleta[:len(conteo_por_estado)])
    ax.set_title("Number of Requests by Status")
    ax.set_xlabel("State")
    ax.set_ylabel("Quantity")
    ax.set_xticklabels(conteo_por_estado.index, rotation=45)
    ax.grid(False)

    # Quitar fondo del gráfico
    fig.patch.set_visible(False)       # Quita fondo general de la figura
    ax.set_facecolor('none')           # Quita fondo del área del gráfico

    plt.tight_layout()
    plt.show()

    # Guardar gráfico
    save_and_close_fig(fig, output_dir, filenames)



    # # GRÁFICO 3: Distribución de tipos de proyecto por cliente

    # # Preparar los datos
    # conteo_cliente_tipo = df.groupby(['client', 'type']).size().unstack().fillna(0)

    # # Crear figura y eje manualmente
    # fig, ax = plt.subplots(figsize=(14, 7))

    # # Dibujar gráfico de barras apiladas
    # conteo_cliente_tipo.plot(
    #     kind='bar',
    #     stacked=True,
    #     ax=ax,
    #     color=mi_paleta[:conteo_cliente_tipo.columns.size]
    # )

    # # Títulos y etiquetas
    # ax.set_title("Distribution of Project Types by Client")
    # ax.set_xlabel("Client")
    # ax.set_ylabel("Request Count")
    # ax.set_xticklabels(conteo_cliente_tipo.index, rotation=90, fontsize=13)
    # ax.tick_params(axis='y', labelsize=13)
    # ax.legend(fontsize=6)

    # # Eliminar fondo
    # fig.patch.set_visible(False)
    # ax.set_facecolor('none')

    # plt.tight_layout()
    # plt.show()

    # # Guardar gráfico
    # save_and_close_fig(fig, output_dir, filenames)



    # # GRÁFICO 4: Tiempo promedio de resolución por cliente

    # # Agrupación
    # df_completadas = df_muestra[df_muestra['status'] == 'Complete'].copy()
    # tiempos_por_cliente = df_completadas.groupby('client')['days_to_complete'].mean().sort_values(ascending=False)

    # # Crear figura y eje
    # fig, ax = plt.subplots(figsize=(12, 6))

    # # Gráfico con Seaborn usando el eje
    # sns.barplot(
    #     x=tiempos_por_cliente.values,
    #     y=tiempos_por_cliente.index,
    #     palette=mi_paleta[:len(tiempos_por_cliente)],
    #     ax=ax
    # )

    # # Personalización
    # ax.set_xlabel("Average Resolution Time (days)")
    # ax.set_ylabel("Client")
    # ax.set_title("Average Resolution Time by Client")
    # ax.tick_params(axis='x', labelrotation=90, labelsize=13)
    # ax.tick_params(axis='y', labelsize=13)
    # ax.grid(False)

    # # Quitar fondo
    # fig.patch.set_visible(False)
    # ax.set_facecolor('none')

    # plt.tight_layout()
    # plt.show()

    # # Guardar gráfico
    # save_and_close_fig(fig, output_dir, filenames)


    return filenames


# if __name__ == "__main__":
#     generate_report()
