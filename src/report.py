import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

import os
# import uuid
from datetime import datetime, timedelta, timezone
import json


# os.chdir(os.path.dirname(__file__))

# with open("issues_data.json", 'r') as file:
#     # Cargar los datos JSON desde el archivo
#     json_data = json.load(file)


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

    # Funcionalidad para manejar archivos .png en un directorio específico:

    # base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    # output_dir = os.path.join(base_dir, 'static', 'reports')
    # os.makedirs(output_dir, exist_ok=True)
    
    # # Elimina imágenes previas del directorio
    # for fname in os.listdir(output_dir):
    #     if fname.endswith(".png"):
    #         os.remove(os.path.join(output_dir, fname))


    df = pd.json_normalize(data)

    json_graphs = []
    

    # GRÁFICOS

    # --- GRÁFICO 1 ---

    df['createdAt'] = pd.to_datetime(df['createdAt'])

    hoy = datetime.now(timezone.utc)
    una_semana_atras = hoy - timedelta(days=7)

    df_semana = df[df['createdAt'] >= una_semana_atras]

    conteo_por_tipo = df_semana['issueType'].value_counts()
    total = conteo_por_tipo.sum()

    labels = conteo_por_tipo.index
    values = conteo_por_tipo.values

    fig_pie = go.Figure(data=[go.Pie(
        labels=labels,
        values=values,
        hole=0.4,
        pull=[0.0] * len(labels),
        marker_colors=mi_paleta[:len(labels)],
        textinfo='percent',
        textfont=dict(size=12, color='black'),
        hoverinfo='label+percent+value'
    )])

    fig_pie.update_layout(
        title_text=f"<b>Distribución de Solicitudes<br>Semana previa al {hoy.strftime('%d/%m/%Y')}<br>Solicitudes recibidas en la última semana: {total}</b>",
        title_x=0.5,
        title_y=1,
        title_font_size=18,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=-0.1,
            xanchor="center",
            x=0.5,
            font=dict(size=20)
        ),
        margin=dict(t=80, b=80, l=40, r=40),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
    )

    # fig_pie.show()

    save_plotly_fig_to_json_list(fig_pie, json_graphs)


    # --- GRÁFICO 2 ---

    np.random.seed(42)
    df['resolutionTime'] = np.random.randint(1, 22, size=len(df))

    resolucion_semanal = df.groupby(df['createdAt'].dt.to_period('W'))['resolutionTime'].mean()

    fig_line = go.Figure()

    fig_line.add_trace(go.Scatter(
        x=resolucion_semanal.index.to_timestamp(), 
        y=resolucion_semanal.values,
        mode='lines+markers',
        marker=dict(color=mi_paleta[1], size=8),
        line=dict(color=mi_paleta[1], width=2),
        name='Tiempo Promedio'
    ))

    fig_line.add_trace(go.Scatter(
        x=resolucion_semanal.index.to_timestamp(),
        y=[21] * len(resolucion_semanal),
        mode='lines',
        line=dict(color='gray', dash='dash', width=1),
        name='Límite SLA'
    ))

    fig_line.update_layout(
        title_text="<b>Evolución del tiempo promedio de resolución semanal</b>",
        title_x=0.5,
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
        hovermode="x unified",
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        ),
        xaxis_showgrid=True,
        yaxis_showgrid=True,
        xaxis_gridcolor='lightgray',
        yaxis_gridcolor='lightgray'
    )

    # fig_line.show()

    save_plotly_fig_to_json_list(fig_line, json_graphs)


    # --- GRÁFICO 3 ---

    fig_client1 = px.histogram(df,
                            x='client.userId',
                            color='issueType',
                            color_discrete_sequence=mi_paleta,
                            title="<b>Distribución de proyectos por cliente</b>")

    fig_client1.update_layout(
        title_x=0.5,
        title_font_size=18,
        xaxis_title="Cliente",
        yaxis_title="Cantidad de solicitudes",
        xaxis=dict(
            title=dict(font=dict(size=16)),
            tickangle=45,
            tickfont=dict(size=13)
        ),
        yaxis=dict(
            title=dict(font=dict(size=16)),
            tickfont=dict(size=13),
            showgrid=True,
            gridcolor='lightgray'
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            title_text="Tipo de Incidencia"
        ),
        bargap=0.1
    )

    # fig_client1.show()

    save_plotly_fig_to_json_list(fig_client1, json_graphs)



    # --- GRÁFICO 4 ---

    df_completadas = df[df['status'] == 'Complete'].copy()

    tiempos_por_cliente = df_completadas.groupby('client.userId')['resolutionTime'].mean().sort_values(ascending=False)

    fig_client2 = px.bar(tiempos_por_cliente,
                        x=tiempos_por_cliente.values,
                        y=tiempos_por_cliente.index,
                        orientation='h',
                        title="<b>Tiempo promedio de resolución por cliente</b>",
                        labels={tiempos_por_cliente.name: "Tiempo promedio de resolución (días)",
                                "index": "Cliente"},
                        color=tiempos_por_cliente.index,
                        color_discrete_sequence=mi_paleta
                        )

    fig_client2.update_yaxes(categoryorder='total ascending')

    fig_client2.update_layout(
        title_x=0.5,
        title_font_size=18,
        xaxis_title="Tiempo promedio de resolución (días)",
        yaxis_title="Cliente",
        xaxis=dict(
            title=dict(font=dict(size=16)),
            tickfont=dict(size=13),
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        yaxis=dict(
            title=dict(font=dict(size=16)),
            tickfont=dict(size=13),
            showgrid=True,
            gridcolor='lightgray',
            gridwidth=1
        ),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)',
        bargap=0.1,
        showlegend=False
    )

    # fig_client2.show()

    save_plotly_fig_to_json_list(fig_client2, json_graphs)
    

    return json_graphs
