import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

import os
import uuid
from datetime import datetime, timedelta, timezone
import json


# from pymongo import MongoClient, DESCENDING


# json_data = """[{"_id":"6847e0955eb9ba10de0cc473","issueType":"Copy revision","status":"In Progress","issueId":"86c3y1ebq","client":{"lockUntil":null,"resetPasswordToken":null,"resetPasswordExpires":null,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"gsyshwn","page":"jsnsh","createdAt":"2025-06-10T07:36:53.707Z","__v":0,"screenshot":"","terraComments":"Muy bien"},{"_id":"6847e11a5eb9ba10de0cc4a7","issueType":"Requested Change","status":"On Hold","issueId":"86c3y1gd5","client":{"lockUntil":null,"resetPasswordToken":null,"resetPasswordExpires":null,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"duask","page":"cnjanl","screenshot":"captura-desde-2025-04-07-17-02-47-250610073906-g3.png","createdAt":"2025-06-10T07:39:06.910Z","__v":0},{"_id":"6847f62a1c4d59011c9f3818","issueType":"Requested Change","status":"Post Launch","issueId":"86c3y4nhz","client":{"lockUntil":null,"resetPasswordToken":null,"resetPasswordExpires":null,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"deed","page":"dedde","screenshot":"captura-desde-2025-04-07-17-02-47-250610090857-d1.png","createdAt":"2025-06-10T09:08:58.584Z","__v":0},{"_id":"68481a2b110e4502d93f4b96","issueType":"Requested Change","status":"On Hold","issueId":"86c3y97z6","client":{"lockUntil":null,"resetPasswordToken":null,"resetPasswordExpires":null,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"deedde","page":"deeded","screenshot":"captura-desde-2025-04-07-17-02-47-250610114234-p9.png","createdAt":"2025-06-10T11:42:35.869Z","__v":0},{"_id":"68481daf18f5b4769c307251","issueType":"Bug Fix","status":"On Hold","issueId":"86c3y9kt3","client":{"lockUntil":null,"resetPasswordToken":null,"resetPasswordExpires":null,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"dede","page":"deeded","createdAt":"2025-06-10T11:57:35.769Z","__v":0},{"_id":"68481dbf18f5b4769c307262","issueType":"Copy revision","status":"Ready to upload","issueId":"86c3y9kya","client":{"lockUntil":null,"resetPasswordToken":null,"resetPasswordExpires":null,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"ededd","page":"ededde","createdAt":"2025-06-10T11:57:51.709Z","__v":0}]"""



mi_paleta = [    
    "#189B5C",  # Emerald Green
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
    """Guarda y cierra una figura de matplotlib."""
    
    filename = f"graph_{uuid.uuid4().hex}.png"
    path = os.path.join(output_dir, filename)

    fig.savefig(path, transparent=True)
    plt.close(fig)

    filenames.append(filename)


def generate_report(data, frequency='weekly'):
    """Genera un informe de gráficos a partir de datos JSON."""

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


    # Cargar datos
    df = pd.json_normalize(data)
    
    # COLUMNAS DEL DATAFRAME:
    #
    # Index(['_id', 'issueType', 'status', 'issueId', 'device', 'browser', 'clientComment', 'page', 'createdAt', '__v', 'screenshot',
    #    'terraComments', 'client.lockUntil', 'client.resetPasswordToken', 'client.resetPasswordExpires', 'client._id', 'client.userId',
    #    'clientComment', 'page', 'createdAt', '__v', 'screenshot', 'terraComments', 'client.lockUntil', 'client.resetPasswordToken',
    #    'client.resetPasswordExpires', 'client._id', 'client.userId', 'client.name', 'client.role', 'client.email', 'client.createdAt',
    #    'terraComments', 'client.lockUntil', 'client.resetPasswordToken', 'client.resetPasswordExpires', 'client._id', 'client.userId',
    #    'client.name', 'client.role', 'client.email', 'client.createdAt', 'client.resetPasswordExpires', 'client._id', 'client.userId',
    #    'client.name', 'client.role', 'client.email', 'client.createdAt', 'client.name', 'client.role', 'client.email', 'client.createdAt',
    #    'client.updatedAt', 'client.__v', 'client.workspaceId', 'client.updatedAt', 'client.__v', 'client.workspaceId',
    #    'client.loginAttempts', 'client.folderId', 'client.spaceId'],
    #   dtype='object')


    filenames = []
    
    # GENERAR GRÁFICOS

    # GRÁFICO 1: Solicitudes recibidas en la última semana
    
    # Convertir 'created_at' a datetime
    df['createdAt'] = pd.to_datetime(df['createdAt'])

    # Calcular la fecha de hace una semana
    hoy = datetime.now(timezone.utc)
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
    ax.set_title(f"Distribución de Solicitudes\nSemana previa al {hoy.strftime('%d/%m/%Y')}\nSolicitudes recibidas en la última semana: {total}", pad=20)
    ax.axis('equal')  # Para que sea un círculo

    # Eliminar fondo
    fig.patch.set_visible(False)
    ax.set_facecolor('none')

    plt.tight_layout()

    # Guardar gráfico
    save_and_close_fig(fig, output_dir, filenames)


    # GRÁFICO 2: Tiempo promedio de resolución semanal

    np.random.seed(42)
    df['resolutionTime'] = np.random.randint(1, 22, size=len(df))

    resolucion_semanal = df.groupby(df['createdAt'].dt.to_period('W'))['resolutionTime'].mean()

    # Crear figura y eje
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.plot(
        resolucion_semanal.index.to_timestamp(),
        resolucion_semanal.values,
        marker='o',
        color=mi_paleta[1]
    )

    ax.set_title("Tiempo promedio de resolución semanal")
    ax.set_xlabel("Semana")
    ax.set_ylabel("Días")
    ax.axhline(21, linestyle='--', color='gray', label='Límite SLA')
    ax.legend()
    ax.grid(True)
    fig.tight_layout()

    # Fondo transparente
    fig.patch.set_visible(False)
    ax.set_facecolor('none')

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

    # # Guardar gráfico
    # save_and_close_fig(fig, output_dir, filenames)

    print(filenames)

    return filenames


# if __name__ == "__main__":
#     generate_report()


# generate_report(json_data)
