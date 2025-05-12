# app/dashboard.py

import mlflow
import pandas as pd
import streamlit as st
import altair as alt

st.set_page_config(page_title="📊 Dashboard General de Evaluación", layout="wide")
st.title("📈 Evaluación Completa del Chatbot por Pregunta")

# ✅ Buscar todos los experimentos que comienzan con "eval_"
client = mlflow.tracking.MlflowClient()
experiments = [exp for exp in client.search_experiments() if exp.name.startswith("eval_")]

if not experiments:
    st.warning("No se encontraron experimentos de evaluación.")
    st.stop()

# Mostrar opciones
exp_names = [exp.name for exp in experiments]
selected_exp_name = st.selectbox("Selecciona un experimento para visualizar:", exp_names)

experiment = client.get_experiment_by_name(selected_exp_name)
runs = client.search_runs(experiment_ids=[experiment.experiment_id], order_by=["start_time DESC"])

if not runs:
    st.warning("No hay ejecuciones registradas en este experimento.")
    st.stop()

# Convertir runs a DataFrame
data = []
for run in runs:
    params = run.data.params
    metrics = run.data.metrics
    data.append({
        "pregunta": params.get("question"),
        "prompt_version": params.get("prompt_version"),
        "chunk_size": int(params.get("chunk_size", 0)),
        "chunk_overlap": int(params.get("chunk_overlap", 0)),
        "correctness": metrics.get("correctness", 0),
        "relevance": metrics.get("relevance", 0),
        "coherence": metrics.get("coherence", 0),
        "toxicity": metrics.get("toxicity", 0),
        "harmfulness": metrics.get("harmfulness", 0),
    })

df = pd.DataFrame(data)

# Mostrar tabla completa
st.subheader("📋 Resultados individuales por pregunta")
st.dataframe(df)

# Agrupación para análisis
grouped = df.groupby(["prompt_version", "chunk_size"]).agg(
    correctness=("correctness", "mean"),
    relevance=("relevance", "mean"),
    coherence=("coherence", "mean"),
    toxicity=("toxicity", "mean"),
    harmfulness=("harmfulness", "mean"),
    preguntas=("pregunta", "count")
).reset_index()

st.subheader("📊 Desempeño promedio de configuración")
st.dataframe(grouped)

# Gráfico
grouped["config"] = grouped["prompt_version"] + " | " + grouped["chunk_size"].astype(str)

# Reformatear el DataFrame a formato largo
melted = grouped.melt(
    id_vars=["config"],
    value_vars=[
        "correctness",
        "relevance",
        "coherence",
        "toxicity",
        "harmfulness"
    ],
    var_name="criterio",
    value_name="promedio"
)

# Gráfico
chart = alt.Chart(melted).mark_bar().encode(
    x=alt.X('criterio:N', title='Criterio', axis=alt.Axis(labelAngle=45)),
    y=alt.Y('promedio:Q', title='Puntaje promedio'),
    color=alt.Color('criterio:N', title='Criterio')
).properties(
    width=700,
    height=500,
    padding={"top": 10, "right": 10, "bottom": 10, "left": 50}
)
st.altair_chart(chart)

# ----------------------------------------------------------------------------------------

# NUEVA SECCIÓN: Análisis de todos los experimentos

# Obtener todos los runs de todos los experimentos
all_runs = []
for exp in experiments:
    runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
    all_runs.extend(runs)

# Convertir todos los runs a DataFrame
data_all = []
for run in all_runs:
    params = run.data.params
    metrics = run.data.metrics
    data_all.append({
        "pregunta": params.get("question"),
        "prompt_version": params.get("prompt_version"),
        "chunk_size": int(params.get("chunk_size", 0)),
        "chunk_overlap": int(params.get("chunk_overlap", 0)),
        "correctness": metrics.get("correctness", 0),
        "relevance": metrics.get("relevance", 0),
        "coherence": metrics.get("coherence", 0),
        "toxicity": metrics.get("toxicity", 0),
        "harmfulness": metrics.get("harmfulness", 0),
    })

df_all = pd.DataFrame(data_all)

# Agrupación para análisis de todos los experimentos
grouped_all = df_all.groupby(["prompt_version", "chunk_size"]).agg(
    correctness=("correctness", "mean"),
    relevance=("relevance", "mean"),
    coherence=("coherence", "mean"),
    toxicity=("toxicity", "mean"),
    harmfulness=("harmfulness", "mean"),
    preguntas=("pregunta", "count")
).reset_index()

st.title("📊 Comparación de todas las configuraciones")
st.dataframe(grouped_all)

# Gráfico para todos los experimentos
grouped_all["config"] = grouped_all["prompt_version"] + " | " + grouped_all["chunk_size"].astype(str)

# ----------------------------------------------------------------------------------------

# Filtrar experimentos por versión
v1_experiments = [exp for exp in experiments if exp.name.endswith("_V1")]
v2_experiments = [exp for exp in experiments if exp.name.endswith("_V2")]

def build_comparison_df(exp_list):
    all_data = []

    # Diccionario para traducir nombres
    nombre_legible = {
        "v1_profesor_estadistica": "Profesor universitario",
        "v2_resumido_directo": "Resumido directo",
        "v3_profesor_primaria": "Profesor de primaria"
    }

    for exp in exp_list:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
        for run in runs:
            params = run.data.params
            metrics = run.data.metrics

            # Extraer prefijo base del nombre
            nombre_raw = exp.name.lower()
            prefijo_legible = next(
                (nombre_legible[prefijo] for prefijo in nombre_legible if nombre_raw.startswith(f"eval_{prefijo}")),
                exp.name  # fallback si no coincide
            )

            all_data.append({
                "experimento": prefijo_legible,
                "correctness": metrics.get("correctness", 0),
                "relevance": metrics.get("relevance", 0),
                "coherence": metrics.get("coherence", 0),
                "toxicity": metrics.get("toxicity", 0),
                "harmfulness": metrics.get("harmfulness", 0),
            })

    df = pd.DataFrame(all_data)
    return df.groupby("experimento").mean().reset_index().melt(
        id_vars=["experimento"],
        var_name="criterio",
        value_name="promedio"
    )

# Gráfica para CHUNK_SIZE=256 y CHUNK_OVERLAP=30
if v1_experiments:
    v1_data = build_comparison_df(v1_experiments)
    st.subheader("📊 Comparación de prompts con CHUNK_SIZE=256 y CHUNK_OVERLAP=30")
    chart_v1 = alt.Chart(v1_data).mark_bar().encode(
        x=alt.X('criterio:N', title='Criterio', axis=alt.Axis(labelAngle=0)),
        xOffset='experimento:N',
        y=alt.Y('promedio:Q', title='Promedio'),
        color=alt.Color('experimento:N', title='Experimento'),
        tooltip=['criterio:N', 'experimento:N', 'promedio:Q']
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(chart_v1, use_container_width=True)
else:
    st.info("No se encontraron experimentos que terminen en _V1.")

# Gráfica para CHUNK_SIZE=512 y CHUNK_OVERLAP=50"
if v2_experiments:
    v2_data = build_comparison_df(v2_experiments)
    st.subheader("📊 Comparación de prompts con CHUNK_SIZE=512 y CHUNK_OVERLAP=50")
    chart_v2 = alt.Chart(v2_data).mark_bar().encode(
        x=alt.X('criterio:N', title='Criterio', axis=alt.Axis(labelAngle=0)),
        xOffset='experimento:N',
        y=alt.Y('promedio:Q', title='Promedio'),
        color=alt.Color('experimento:N', title='Experimento'),
        tooltip=['criterio:N', 'experimento:N', 'promedio:Q']
    ).properties(
        width=700,
        height=400
    )
    st.altair_chart(chart_v2, use_container_width=True)
else:
    st.info("No se encontraron experimentos que terminen en V2.")

# ----------------------------------------------------------------------------------------

v1_prefijo = [exp for exp in experiments if exp.name.lower().startswith("eval_v1_profesor_estadistica")]
v2_prefijo = [exp for exp in experiments if exp.name.lower().startswith("eval_v2_resumido_directo")]
v3_prefijo = [exp for exp in experiments if exp.name.lower().startswith("eval_v3_profesor_primaria")]

def build_comparison_df_vectorstore(exp_list):
    all_data = []
    for exp in exp_list:
        runs = client.search_runs(experiment_ids=[exp.experiment_id], order_by=["start_time DESC"])
        for run in runs:
            params = run.data.params
            metrics = run.data.metrics

            # Asignar alias por sufijo
            if exp.name.strip().upper().endswith("V1"):
                label = "Size=256, overlap=30"
            elif exp.name.strip().upper().endswith("V2"):
                label = "Size=512, overlap=50"
            else:
                label = exp.name  # fallback

            all_data.append({
                "experimento": label,
                "correctness": metrics.get("correctness", 0),
                "relevance": metrics.get("relevance", 0),
                "coherence": metrics.get("coherence", 0),
                "toxicity": metrics.get("toxicity", 0),
                "harmfulness": metrics.get("harmfulness", 0),
            })

    df = pd.DataFrame(all_data)

    # Agrupar y reformatear
    return df.groupby("experimento").mean().reset_index().melt(
        id_vars=["experimento"],
        var_name="criterio",
        value_name="promedio"
    )


# Gráfica para v1_profesor_estadistica
if v1_prefijo:
    v1_df = build_comparison_df_vectorstore(v1_prefijo)
    st.subheader("📊 Comparación con prompt v1_profesor_estadistica")
    chart_v1_pref = alt.Chart(v1_df).mark_bar().encode(
        x=alt.X('criterio:N', title='Criterio', axis=alt.Axis(labelAngle=0)),
        xOffset='experimento:N',
        y=alt.Y('promedio:Q', title='Promedio'),
        color=alt.Color('experimento:N', title='Experimento'),
        tooltip=['criterio:N', 'experimento:N', 'promedio:Q']
    ).properties(width=700, height=400)
    st.altair_chart(chart_v1_pref, use_container_width=True)
else:
    st.info("No se encontraron experimentos que empiecen con v1_profesor_estadistica.")

# Gráfica para v2_resumido_directo
if v2_prefijo:
    v2_df = build_comparison_df_vectorstore(v2_prefijo)
    st.subheader("📊 Comparación con prompt v2_resumido_directo")
    chart_v2_pref = alt.Chart(v2_df).mark_bar().encode(
        x=alt.X('criterio:N', title='Criterio', axis=alt.Axis(labelAngle=0)),
        xOffset='experimento:N',
        y=alt.Y('promedio:Q', title='Promedio'),
        color=alt.Color('experimento:N', title='Experimento'),
        tooltip=['criterio:N', 'experimento:N', 'promedio:Q']
    ).properties(width=700, height=400)
    st.altair_chart(chart_v2_pref, use_container_width=True)
else:
    st.info("No se encontraron experimentos que empiecen con v2_resumido_directo.")

# Gráfica para v3_profesor_primaria
if v3_prefijo:
    v3_df = build_comparison_df_vectorstore(v3_prefijo)
    st.subheader("📊 Comparación con prompt v3_profesor_primaria")
    chart_v3_pref = alt.Chart(v3_df).mark_bar().encode(
        x=alt.X('criterio:N', title='Criterio', axis=alt.Axis(labelAngle=0)),
        xOffset='experimento:N',
        y=alt.Y('promedio:Q', title='Promedio'),
        color=alt.Color('experimento:N', title='Experimento'),
        tooltip=['criterio:N', 'experimento:N', 'promedio:Q']
    ).properties(width=700, height=400)
    st.altair_chart(chart_v3_pref, use_container_width=True)
else:
    st.info("No se encontraron experimentos que empiecen con v3_profesor_primaria.")