import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import plotly.graph_objects as go


import os
import uuid
from datetime import datetime, timedelta, timezone
import json


# from pymongo import MongoClient, DESCENDING


json_data = [{"_id":"6847e0955eb9ba10de0cc473","issueType":"Copy revision","status":"In Progress","issueId":"86c3y1ebq","client":{"lockUntil":None,"resetPasswordToken":None,"resetPasswordExpires":None,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"gsyshwn","page":"jsnsh","createdAt":"2025-06-10T07:36:53.707Z","__v":0,"screenshot":"","terraComments":"Muy bien"},{"_id":"6847e11a5eb9ba10de0cc4a7","issueType":"Requested Change","status":"On Hold","issueId":"86c3y1gd5","client":{"lockUntil":None,"resetPasswordToken":None,"resetPasswordExpires":None,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"duask","page":"cnjanl","screenshot":"captura-desde-2025-04-07-17-02-47-250610073906-g3.png","createdAt":"2025-06-10T07:39:06.910Z","__v":0},{"_id":"6847f62a1c4d59011c9f3818","issueType":"Requested Change","status":"Post Launch","issueId":"86c3y4nhz","client":{"lockUntil":None,"resetPasswordToken":None,"resetPasswordExpires":None,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"deed","page":"dedde","screenshot":"captura-desde-2025-04-07-17-02-47-250610090857-d1.png","createdAt":"2025-06-10T09:08:58.584Z","__v":0},{"_id":"68481a2b110e4502d93f4b96","issueType":"Requested Change","status":"On Hold","issueId":"86c3y97z6","client":{"lockUntil":None,"resetPasswordToken":None,"resetPasswordExpires":None,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"deedde","page":"deeded","screenshot":"captura-desde-2025-04-07-17-02-47-250610114234-p9.png","createdAt":"2025-06-10T11:42:35.869Z","__v":0},{"_id":"68481daf18f5b4769c307251","issueType":"Bug Fix","status":"On Hold","issueId":"86c3y9kt3","client":{"lockUntil":None,"resetPasswordToken":None,"resetPasswordExpires":None,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"dede","page":"deeded","createdAt":"2025-06-10T11:57:35.769Z","__v":0},{"_id":"68481dbf18f5b4769c307262","issueType":"Copy revision","status":"Ready to upload","issueId":"86c3y9kya","client":{"lockUntil":None,"resetPasswordToken":None,"resetPasswordExpires":None,"_id":"6846f64b15f6e0aa2607acae","userId":"1RHPA8","name":"Igor Aparicio","role":"admin","email":"igoruve@gmail.com","createdAt":"2025-06-09T14:57:15.535Z","updatedAt":"2025-06-10T13:26:04.381Z","__v":0,"workspaceId":"90151243006","loginAttempts":0,"folderId":"90157537518","spaceId":"90155116219"},"device":"Desktop","browser":"Google Chrome","clientComment":"ededd","page":"ededde","createdAt":"2025-06-10T11:57:51.709Z","__v":0}]


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


def save_plotly_fig_to_json_list(plotly_fig_obj, json_graphs):
    chart_json_string = plotly_fig_obj.to_json()
    
    json_graphs.append(chart_json_string)


def generate_report(data, frequency='weekly'):
    """Genera un informe de gráficos a partir de datos JSON."""

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    output_dir = os.path.join(base_dir, 'static', 'reports')
    os.makedirs(output_dir, exist_ok=True)
    
    # Elimina imágenes previas del directorio
    for fname in os.listdir(output_dir):
        if fname.endswith(".png"):
            os.remove(os.path.join(output_dir, fname))


    # Cargar datos
    df = pd.json_normalize(data)

    json_graphs = []
    
    # GENERAR GRÁFICOS

    # --- GRÁFICO 1 ---

    # Convertir 'createdAt' a datetime
    df['createdAt'] = pd.to_datetime(df['createdAt'])

    # Calcular la fecha de hace una semana (en UTC)
    hoy = datetime.now(timezone.utc)
    una_semana_atras = hoy - timedelta(days=7)

    # Filtrar las solicitudes de la última semana
    df_semana = df[df['createdAt'] >= una_semana_atras]

    # Contar incidencias por tipo
    conteo_por_tipo = df_semana['issueType'].value_counts()
    total = conteo_por_tipo.sum()

    # Preparar los datos para Plotly
    labels = conteo_por_tipo.index
    values = conteo_por_tipo.values

    # Crear el gráfico de donut de Plotly
    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4, # Crea un gráfico de donut
        pull=[0.05] * len(labels), # Separa ligeramente las porciones para destacarlas
        marker_colors=mi_paleta[:len(labels)], # Aplica los colores
        textinfo='percent', # Muestra el porcentaje directamente en las porciones
        textfont=dict(size=12, color='black'), # Estilo del texto del porcentaje
        hoverinfo='label+percent+value' # Información que aparece al pasar el ratón
    )])

    # Configurar el diseño del gráfico y el título
    fig_pie.update_layout(
        title_text=f"<b>Distribución de Solicitudes</b><br>Semana previa al {hoy.strftime('%d/%m/%Y')}<br>Solicitudes recibidas en la última semana: {total}",
        title_x=0.5, # Centra el título
        title_font_size=16,
        legend=dict(
            orientation="h", # Leyenda horizontal
            yanchor="bottom",
            y=-0.2, # Posiciona la leyenda debajo del gráfico
            xanchor="center",
            x=0.5
        ),
        margin=dict(t=80, b=80, l=40, r=40), # Ajusta los márgenes
        paper_bgcolor='rgba(0,0,0,0)', # Fondo del papel (fuera del área del gráfico) transparente
        plot_bgcolor='rgba(0,0,0,0)', # Fondo del área del gráfico transparente
    )

    fig_pie.show() # ¡Esta es la línea clave para visualizar!

    save_plotly_fig_to_json_list(fig_pie, json_graphs)

    # print(fig_pie.to_json())


    # --- GRÁFICO 2 ---

    np.random.seed(42)
    df['resolutionTime'] = np.random.randint(1, 22, size=len(df))

    resolucion_semanal = df.groupby(df['createdAt'].dt.to_period('W'))['resolutionTime'].mean()

    print(resolucion_semanal)

    # Crear el gráfico de línea de Plotly
    fig_line = go.Figure()

    # Añadir la línea principal del promedio de resolución
    fig_line.add_trace(go.Scatter(
        x=resolucion_semanal.index.to_timestamp(), # Eje X: semanas convertidas a timestamp
        y=resolucion_semanal.values, # Eje Y: valores de resolución
        mode='lines+markers', # Muestra tanto líneas como marcadores
        marker=dict(color=mi_paleta[1], size=8), # Color del marcador de tu paleta
        line=dict(color=mi_paleta[1], width=2), # Color y grosor de la línea
        name='Tiempo Promedio'
    ))

    # Añadir la línea del Límite SLA (línea horizontal)
    fig_line.add_trace(go.Scatter(
        x=resolucion_semanal.index.to_timestamp(), # Usa las mismas semanas para la línea SLA
        y=[21] * len(resolucion_semanal), # Valor fijo de 21 para cada punto
        mode='lines',
        line=dict(color='gray', dash='dash', width=1), # Estilo de línea punteada
        name='Límite SLA'
    ))

    # Configurar el diseño y los títulos del gráfico
    fig_line.update_layout(
        title_text="<b>Tiempo promedio de resolución semanal</b>",
        title_x=0.5, # Centra el título
        title_font_size=16,
        xaxis_title="Semana",
        yaxis_title="Días",
        hovermode="x unified", # Muestra tooltip para todos los trazos en un punto X
        # Eliminar fondo (transparente)
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        # Configuración de la leyenda
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02, # Coloca la leyenda encima del gráfico
            xanchor="right",
            x=1
        ),
        # Mostrar cuadrícula
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridcolor='lightgray',
        yaxis_gridcolor='lightgray'
    )

    fig_line.show() # ¡Esta es la línea clave para visualizar!

    save_plotly_fig_to_json_list(fig_pie, json_graphs)

    # print(fig_pie.to_json())



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

    # print(filenames)

    return json_graphs


generate_report(json_data)
