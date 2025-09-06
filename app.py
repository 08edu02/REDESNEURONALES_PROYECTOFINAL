import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import wfdb
import neurokit2 as nk
import os
from pathlib import Path
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# --- Configuración de la Página ---
st.set_page_config(
    page_title="Análisis de ECG",
    page_icon="❤️",
    layout="wide"
)

# --- Título y Descripción ---
st.title("Análisis y Visualización de Electrocardiogramas (ECG)")
st.markdown("""
Esta aplicación permite visualizar señales de ECG de 12 derivaciones, analizar la frecuencia cardiaca 
y (opcionalmente) OOOO clasificar el ritmo cardiaco utilizando un modelo de Deep Learning.
""")

# --- Funciones de Carga y Procesamiento ---

@st.cache_data
def load_record(record_path):
    """
    Carga la señal y metadatos de un registro desde una ruta local.
    """
    try:
        record = wfdb.rdrecord(record_path)
        st.success(f"Registro '{record_path}' cargado exitosamente.")
        return record
    except Exception as e:
        st.error(f"No se pudo cargar el registro desde '{record_path}'. Error: {e}")
        return None

@st.cache_resource
def load_classification_model():
    """
    Carga el modelo de clasificación de Keras pre-entrenado.
    """
    model_path = 'ecg_classifier.h5'
    if os.path.exists(model_path):
        model = load_model(model_path)
        return model
    return None

# Función para encontrar registros dinámicamente
@st.cache_data
def find_local_records(root_dir="data/WFDBRecords"):
    """
    Escanea el directorio raíz de forma recursiva para encontrar todos los registros de ECG (.hea).
    Retorna un diccionario con nombres amigables como llaves y rutas de archivo como valores.
    """
    root_path = Path(root_dir)
    if not root_path.is_dir():
        st.error(f"El directorio de datos '{root_dir}' no fue encontrado. Por favor, asegúrese de que exista.")
        return {}

    # Busca todos los archivos .hea en todos los subdirectorios
    hea_files = sorted(list(root_path.glob('**/*.hea')))
    
    records_dict = {}
    for file in hea_files:
        # La ruta que necesita wfdb (sin la extensión .hea)
        # Ejemplo: 'data/WFDBRecords/01/010/JS00001'
        file_path_without_ext = str(file.with_suffix(''))
        
        # El nombre que mostraremos en el selectbox (ruta relativa sin extensión)
        # Ejemplo: '01/010/JS00001'
        display_name = str(file.relative_to(root_path).with_suffix(''))
        
        records_dict[display_name] = file_path_without_ext
        
    return records_dict

@st.cache_data
def load_snomed_definitions(file_path="data/ConditionNames_SNOMED-CT.csv"):
    """
    Carga las definiciones de los códigos SNOMED CT desde el archivo CSV.
    Retorna un diccionario para una búsqueda rápida de código -> definición.
    """
    try:
        df = pd.read_csv(file_path)
        
        # 'Full Name' para las descripciones y 'Snomed_CT' para los códigos
        return pd.Series(df['Full Name'].values, index=df['Snomed_CT']).to_dict()

    except FileNotFoundError:
        st.warning(f"El archivo de definiciones SNOMED '{file_path}' no fue encontrado. Las descripciones no estarán disponibles.")
        return {}
    except KeyError:
        st.error(f"El archivo '{file_path}' no tiene las columnas esperadas ('Snomed_CT', 'Full Name'). No se pueden cargar las definiciones.")
        return {}

# --- Funciones de Visualización ---

def plot_ecg_professional_plotly(signal_data, metadata):
    fs = metadata['fs']
    time = np.arange(signal_data.shape[0]) / fs
    num_leads = len(metadata['sig_name'])
    
    # AUMENTAR ESPACIADO VERTICAL Y HORIZONTAL
    fig = make_subplots(
        rows=6, 
        cols=2, 
        subplot_titles=[f"Derivación: {name}" for name in metadata['sig_name']],
        vertical_spacing=0.08,  # Aumentar espacio vertical entre filas
        horizontal_spacing=0.1   # Aumentar espacio horizontal entre columnas
    )
    
    for i, lead_name in enumerate(metadata['sig_name']):
        row = i // 2 + 1
        col = i % 2 + 1
        signal = signal_data[:, i]
        
        fig.add_trace(
            go.Scatter(x=time, y=signal, mode='lines', line=dict(color='black', width=1)),
            row=row, col=col
        )
        
        # Configuración estilo papel ECG
        fig.update_xaxes(
            showgrid=True, gridwidth=1, gridcolor='lightgray',
            dtick=0.2, minor=dict(dtick=0.04, gridcolor='rgba(0,0,0,0.1)', showgrid=True),
            row=row, col=col,
            range=[0, 10]  # Limitar a 10 segundos
        )
        fig.update_yaxes(
            showgrid=True, gridwidth=1, gridcolor='lightgray', 
            dtick=0.5, minor=dict(dtick=0.1, gridcolor='rgba(0,0,0,0.1)', showgrid=True),
            row=row, col=col
        )
    
    # ✅ AUMENTAR ALTURA TOTAL y ajustar márgenes
    fig.update_layout(
        height=1500,  # Aumentar altura
        showlegend=False,
        title_text="ECG - 12 Derivaciones (Estilo Papel Electrocardiográfico)",
        margin=dict(l=50, r=50, t=80, b=50),  # Márgenes ajustados
    )
    
    return fig

"""
Grafica una derivación de ECG (usualmente la II) y marca los picos R detectados
"""
def plot_ecg_with_peaks(signal_data, metadata, rpeaks):
    # Esta función crea un gráfico interactivo de una sola derivación de ECG
    # Parámetros:
    #   - signal_data: El array completo de señales (12 derivaciones)
    #   - metadata: El diccionario con la información del registro (ej. 'fs', 'sig_name')
    #   - rpeaks: Un array de NumPy con los índices de los picos R detectados
    
    fs = metadata['fs']
    time = np.arange(signal_data.shape[0]) / fs
    
    try:
        lead_index = metadata['sig_name'].index('II')
        lead_name = 'II'
    except ValueError:
        lead_index = 0
        lead_name = metadata['sig_name'][0]
        
    signal = signal_data[:, lead_index]
    
    # Usando Plotly para consistencia visual
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time, y=signal, mode='lines', name=f'Señal (Derivación {lead_name})', line=dict(color='blue')))

    # Ahora los marcadores solo se añaden si la lista de picos no está vacía
    if len(rpeaks) > 0:
        fig.add_trace(go.Scatter(x=time[rpeaks], y=signal[rpeaks], mode='markers', name='Picos R', marker=dict(color='red', size=10, symbol='x')))
    
    fig.update_layout(
        title="Detección de Picos R para Análisis de Frecuencia Cardiaca",
        xaxis_title="Tiempo (s)",
        yaxis_title="Voltaje (mV)",
        legend_title="Leyenda"
    )
    return fig

def scroll_to_top():
    """
    Añade un botón flotante en la esquina inferior derecha que, al hacer clic,
    desplaza la página suavemente hacia arriba.
    """
    # Esta función inyecta código HTML y CSS en la aplicación de Streamlit
    # para crear un botón de "volver arriba".
    
    # CSS para el estilo y posicionamiento del botón
    button_style = """
        <style>
            /* Establece el comportamiento de scroll suave para toda la página */
            html {
                scroll-behavior: smooth;
            }
            /* Estilo del botón flotante */
            #scrollTopBtn {
                position: fixed; /* Fijo en la pantalla */
                bottom: 20px; /* Distancia desde abajo */
                right: 20px; /* Distancia desde la derecha */
                z-index: 999; /* Asegura que esté por encima de otros elementos */
                border: none;
                outline: none;
                background-color: #007bff; /* Color de fondo azul */
                color: white; /* Color del ícono */
                cursor: pointer;
                padding: 15px;
                border-radius: 50%; /* Forma circular */
                font-size: 18px;
                box-shadow: 0px 4px 8px rgba(0, 0, 0, 0.2);
                transition: background-color 0.3s, transform 0.3s;
            }
            #scrollTopBtn:hover {
                background-color: #0056b3; /* Azul más oscuro al pasar el cursor */
                transform: scale(1.1); /* Ligeramente más grande al pasar el cursor */
            }
        </style>
    """
    
    # HTML para el botón. El href="#" es un ancla que apunta al inicio de la página.
    button_html = '<a href="#" id="scrollTopBtn">⬆️</a>'
    
    # Combina el CSS y el HTML y lo muestra en la app
    st.markdown(button_style + button_html, unsafe_allow_html=True)

# --- Barra Lateral (Sidebar) para Controles ---
st.sidebar.header("Panel de Control")

registros = find_local_records()

if not registros:
    st.sidebar.error("No se encontraron registros en la carpeta 'data/WFDBRecords'.")
    st.stop() # Detiene la ejecución si no hay datos

selected_key = st.sidebar.selectbox(
    "📋 Seleccionar registro:",
    options=list(registros.keys()),
    index=0,
    help="Seleccione un registro de ECG de la lista de archivos disponibles."
)

record_path = registros[selected_key]

st.sidebar.divider()
st.sidebar.header("Integrantes del Grupo")
st.sidebar.markdown("""
- Julio Barrios
- Carlos Carranza
- Carlos Chavarria
- Giancarlo Lamadrid
""")


# --- Cuerpo Principal de la Aplicación ---
record = load_record(record_path)

# Carga las definiciones de SNOMED una sola vez
snomed_definitions = load_snomed_definitions()

if record:
    signal_mv = record.p_signal
    if record.units[0].lower() == 'uv':
        signal_mv = signal_mv / 1000.0

    with st.expander("Ver Metadatos del Registro", expanded=False):
        st.write(f"**Nombre del Registro:** {record.record_name}")
        st.write(f"**Frecuencia de Muestreo:** {record.fs} Hz")
        st.write(f"**Duración de la Señal:** {record.sig_len / record.fs} segundos")
        st.write(f"**Número de Derivaciones:** {record.n_sig}")
        st.write(f"**Nombres de Derivaciones:** {', '.join(record.sig_name)}")
        st.write(f"**Unidades:** {', '.join(record.units)}")
        st.write("**Comentarios/Diagnóstico:**")
        for comment in record.comments:
            clean_comment = comment.replace('Unknown', 'Desconocido')
            st.text(f"- {clean_comment}")

            # Si la línea es un diagnóstico y tenemos definiciones cargadas...
            if clean_comment.startswith('Dx:') and snomed_definitions:
                # Extrae los códigos (ej. '426783006,251146004')
                codes_str = clean_comment.split(':')[1].strip()
                codes = [int(c.strip()) for c in codes_str.split(',')]
                
                # Muestra cada código con su definición
                for code in codes:
                    # .get() es una forma segura de buscar, devuelve un texto por defecto si no encuentra el código
                    definition = snomed_definitions.get(code, "Definición no encontrada en el archivo local.")
                    st.markdown(f"  - **`{code}`**: *{definition}*")

        st.info("""
        **Glosario de Términos:**
        - **Dx:** Diagnóstico principal.
        - **Rx:** Medicación (Terapia farmacológica).
        - **Hx:** Historia clínica del paciente.
        - **Sx:** Síntomas reportados.
        - **Derivadas de las extremidades:**
            - **aVR:** Aumentada Vector Derecha (Augmented Vector Right).
            - **aVL:** Aumentada Vector Izquierda (Augmented Vector Left).
            - **aVF:** Aumentada Vector Pie (Augmented Vector Foot).
        """)

    st.header("Objetivo 1: Visualización de ECG (12 Derivaciones)")
    st.markdown("Gráfico interactivo de las 12 derivaciones, simulando el formato de papel electrocardiográfico estándar (25 mm/s, 10 mm/mV).")
    fig_professional = plot_ecg_professional_plotly(signal_mv, record.__dict__)
    st.plotly_chart(fig_professional, use_container_width=True)

    st.header("Objetivo 2: Análisis de Frecuencia Cardiaca")
    st.markdown("Se utiliza la librería `neurokit2` para detectar los picos R en la derivación II y calcular la frecuencia cardiaca (FC). Se emite una alerta si la FC promedio está fuera del rango normal (60-100 lpm).")
    
    # Seleccionar la deriv II para el análisis
    try:
        lead_ii_index = record.sig_name.index('II')
    except ValueError:
        st.warning("No se encontró la derivación 'II'. Usando la primera derivación para el análisis.")
        lead_ii_index = 0
        
    ecg_signal_for_analysis = signal_mv[:, lead_ii_index]
    
    # Procesar la señal con NeuroKit2
    _, rpeaks = nk.ecg_peaks(ecg_signal_for_analysis, sampling_rate=record.fs)

    r_peaks_indices = rpeaks['ECG_R_Peaks']

    col1, col2 = st.columns([1, 2]) # Dar más espacio al gráfico

    help_text_picos_r = """
    Los picos R son los puntos más altos del complejo QRS en un ECG. 
    Representan la contracción de los ventrículos (las cámaras principales del corazón). 
    La frecuencia cardiaca se calcula midiendo el tiempo entre picos R consecutivos.
    """

    if len(r_peaks_indices) > 0:
        # Si hay picos, calcula la FC y muestra la métrica y la alerta
        heart_rate = nk.ecg_rate(r_peaks_indices, sampling_rate=record.fs, desired_length=len(ecg_signal_for_analysis))
        avg_heart_rate = np.mean(heart_rate)
    
        with col1:
            st.metric(label="Frecuencia Cardiaca Promedio", value=f"{avg_heart_rate:.2f} lpm", help=help_text_picos_r)
            if 60 <= avg_heart_rate <= 100:
                st.success("La frecuencia cardiaca está en el rango normal.")
            else:
                st.error("¡Alerta! La frecuencia cardiaca está fuera del rango normal.")
    else:
        # Si no hay picos, muestra la advertencia
        with col1:
            st.metric(label="Frecuencia Cardiaca Promedio", value="N/A", help=help_text_picos_r)
            st.warning("No se pudieron detectar picos R en esta señal.")

    # Mostramos el gráfico en cualquier caso
    with col2:
        st.write("Visualización de Picos R:")
        fig_peaks = plot_ecg_with_peaks(signal_mv, record.__dict__, r_peaks_indices)
        st.plotly_chart(fig_peaks, use_container_width=True)

    st.header("Objetivo 3 (Opcional): Clasificación de Arritmia con Red Neuronal")
    model = load_classification_model()
    if model is None:
        st.warning("""
        **Funcionalidad no disponible.** Para activar la clasificación, debe entrenar un modelo de red neuronal
        y guardarlo como `ecg_classifier.h5` en la misma carpeta que la aplicación.
        """)
    else:
        class_names = ['Sinus Bradycardia', 'Sinus Rhythm', 'Atrial Fibrillation', 'Sinus Tachycardia']
        if st.button("Clasificar Ritmo Cardiaco"):
            with st.spinner("El modelo está analizando la señal..."):
                # Preparar la señal para el modelo (usar la misma derivación y asegurar el tamaño correcto)
                signal_to_classify = ecg_signal_for_analysis
                signal_reshaped = np.expand_dims(np.expand_dims(signal_to_classify, axis=0), axis=-1)
                prediction = model.predict(signal_reshaped)
                predicted_class_index = np.argmax(prediction)
                predicted_class_name = class_names[predicted_class_index]
                confidence = np.max(prediction) * 100

                st.subheader(f"Resultado de la Clasificación: **{predicted_class_name}**")
                st.write(f"Confianza del modelo: **{confidence:.2f}%**")

                # Mostrar probabilidades de todas las clases
                probs_df = pd.DataFrame(prediction, columns=class_names, index=["Probabilidad"])
                st.dataframe(probs_df.style.format("{:.2%}"))
else:
    st.info("Por favor, seleccione un registro válido en la barra lateral.")

scroll_to_top()