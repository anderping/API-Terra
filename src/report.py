import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
# import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go

import os
# import uuid
from datetime import datetime, timedelta, timezone
import json


# from pymongo import MongoClient, DESCENDING

os.chdir(os.path.dirname(__file__))

with open("issues_data.json", 'r') as file:
    # Cargar los datos JSON desde el archivo
    json_data = json.load(file)


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
    
    chart_dict = json.loads(chart_json_string)

    json_graphs.append(chart_dict)


def generate_report(data, frequency='weekly'):
    """Genera un informe de gráficos a partir de datos JSON."""

    # base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # output_dir = os.path.join(base_dir, 'static', 'reports')
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Elimina imágenes previas del directorio
    # for fname in os.listdir(output_dir):
    #     if fname.endswith(".png"):
    #         os.remove(os.path.join(output_dir, fname))


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
        pull=[0.0] * len(labels), # Separa ligeramente las porciones para destacarlas
        marker_colors=mi_paleta[:len(labels)], # Aplica los colores
        textinfo='percent', # Muestra el porcentaje directamente en las porciones
        textfont=dict(size=12, color='black'), # Estilo del texto del porcentaje
        hoverinfo='label+percent+value' # Información que aparece al pasar el ratón
    )])

    # Configurar el diseño del gráfico y el título
    fig_pie.update_layout(
        title_text=f"<b>Distribución de Solicitudes<br>Semana previa al {hoy.strftime('%d/%m/%Y')}<br>Solicitudes recibidas en la última semana: {total}</b>",
        title_x=0.5, # Centra el título
        title_y=1, # Ajusta la posición vertical del título
        title_font_size=18,
        legend=dict(
            orientation="h", # Leyenda horizontal
            yanchor="bottom",
            y=-0.1, # Posiciona la leyenda debajo del gráfico
            xanchor="center",
            x=0.5,
            font=dict(size=20)
        ),
        margin=dict(t=80, b=80, l=40, r=40), # Ajusta los márgenes
        paper_bgcolor='rgba(0,0,0,0)', # Fondo del papel (fuera del área del gráfico) transparente
        plot_bgcolor='rgba(0,0,0,0)', # Fondo del área del gráfico transparente
    )

    fig_pie.show()

    save_plotly_fig_to_json_list(fig_pie, json_graphs)

    # print(fig_pie.to_json())


    # --- GRÁFICO 2 ---

    np.random.seed(42)
    df['resolutionTime'] = np.random.randint(1, 22, size=len(df))

    resolucion_semanal = df.groupby(df['createdAt'].dt.to_period('W'))['resolutionTime'].mean()

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
        title_text="<b>Evolución del tiempo promedio de resolución semanal</b>",
        title_x=0.5, # Centra el título
        title_font_size=18,
        xaxis_title="Semana",
        yaxis_title="Días",
        xaxis=dict(
            title=dict(
                font=dict(
                    size=16
                )
            )
        ),
        yaxis=dict(
            title=dict(
                font=dict(
                    size=16
                )
            )
        ),
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

    fig_line.show()

    save_plotly_fig_to_json_list(fig_line, json_graphs)

    # print(fig_pie.to_json())



    # --- GRÁFICO 3: Distribución de tipos de proyecto por cliente ---

    fig_client1 = px.histogram(df,
                            x='client.userId', # Usamos 'client.userId' como columna para el eje X
                            color='issueType',
                            color_discrete_sequence=mi_paleta, # ¡Aquí se usa tu paleta!
                            title="<b>Distribución de proyectos por cliente</b>")

    # Configurar el diseño del gráfico para mayor claridad y estética
    fig_client1.update_layout(
        title_x=0.5, # Centra el título
        title_font_size=18,
        xaxis_title="Cliente",
        yaxis_title="Cantidad de solicitudes",
        xaxis=dict(
            title=dict(font=dict(size=16)),
            tickangle=45, # Inclina las etiquetas del eje X para que no se solapen
            tickfont=dict(size=13)
        ),
        yaxis=dict(
            title=dict(font=dict(size=16)),
            tickfont=dict(size=13),
            showgrid=True, # Mostrar cuadrícula en eje Y
            gridcolor='lightgray'
        ),
        paper_bgcolor='rgba(0,0,0,0)', # Fondo transparente del papel del gráfico
        plot_bgcolor='rgba(0,0,0,0)', # Fondo transparente del área de trazado
        legend=dict(
            orientation="h", # Leyenda horizontal
            yanchor="bottom",
            y=1.02, # Posición de la leyenda (encima del gráfico)
            xanchor="right",
            x=1,
            title_text="Tipo de Incidencia"
        ),
        bargap=0.1 # Espacio entre las barras
    )

    # Mostrar el gráfico (en un entorno interactivo como Jupyter o un script que lo abra en el navegador)
    fig_client1.show()

    save_plotly_fig_to_json_list(fig_client1, json_graphs)



    # GRÁFICO 4: Tiempo promedio de resolución por cliente

    df_completadas = df[df['status'] == 'Complete'].copy()

    tiempos_por_cliente = df_completadas.groupby('client.userId')['resolutionTime'].mean().sort_values(ascending=False)

    # Crear el gráfico de barras horizontal con Plotly Express
    fig_client2 = px.bar(tiempos_por_cliente,
                        x=tiempos_por_cliente.values,
                        y=tiempos_por_cliente.index,
                        orientation='h', # Define el gráfico como horizontal
                        title="<b>Tiempo promedio de resolución por cliente</b>",
                        labels={tiempos_por_cliente.name: "Tiempo promedio de resolución (días)",
                                "index": "Cliente"}, # Personaliza las etiquetas de los ejes
                        color=tiempos_por_cliente.index, # Colorea las barras según el 'userID' (categoría)
                        color_discrete_sequence=mi_paleta # ¡Aquí se usa tu paleta discreta!
                        )

    # Ordenar las categorías del eje Y para que el valor más alto (mayor tiempo) esté arriba
    fig_client2.update_yaxes(categoryorder='total ascending') # 'total ascending' ordena por la longitud de la barra en gráficos horizontales

    # Personalización del diseño del gráfico
    fig_client2.update_layout(
        title_x=0.5, # Centra el título
        title_font_size=18,
        xaxis_title="Tiempo promedio de resolución (días)",
        yaxis_title="Cliente",
        xaxis=dict(
            title=dict(font=dict(size=16)),
            tickfont=dict(size=13),
            showgrid=True, # Mostrar cuadrícula en el eje X
            gridcolor='lightgray', # Color de la cuadrícula
            gridwidth=1 # Ancho de la línea de la cuadrícula
        ),
        yaxis=dict(
            title=dict(font=dict(size=16)),
            tickfont=dict(size=13),
            showgrid=True, # Mostrar cuadrícula en el eje Y
            gridcolor='lightgray', # Color de la cuadrícula
            gridwidth=1 # Ancho de la línea de la cuadrícula
        ),
        paper_bgcolor='rgba(0,0,0,0)', # Fondo transparente del papel del gráfico
        plot_bgcolor='rgba(0,0,0,0)', # Fondo transparente del área de trazado
        bargap=0.1, # Espacio entre las barras
        showlegend=False # Oculta la leyenda de colores si los colores representan las categorías de los clientes y no un mapeo continuo.
    )

    # Mostrar el gráfico (esto abrirá una ventana o lo mostrará en tu entorno Jupyter)
    fig_client2.show()

    save_plotly_fig_to_json_list(fig_client2, json_graphs)


    # for i, chart_json_string in enumerate(json_graphs):   
    #     # Paso 1: Convertir la cadena JSON a un diccionario de Python
    #     chart_dict = json.loads(chart_json_string)
        
    #     # Paso 2: Crear el objeto Figure de Plotly a partir del diccionario
    #     fig_reconstructed = go.Figure(chart_dict)
        
    #     # Paso 3: Visualizar el gráfico
    #     fig_reconstructed.show()

    return json_graphs


generate_report(json_data)

