import streamlit as st
import streamlit_shadcn_ui as ui
import Getdata
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import plotly.express as px
import plotly.graph_objects as go
import matplotlib as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.ensemble import RandomForestClassifier 
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, precision_score, recall_score, f1_score
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc




st.sidebar.title("Regresión Logística")

if "opcion" not in st.session_state:
    st.session_state.opcion = "Informacion"

with st.sidebar:
    if ui.button(text="Ejemplos", key="btn2", className="text-white w-full"):
        st.session_state.opcion = "Ejemplos"
    if ui.button(text="Calculadora", key="btn3", className="text-white w-full"):
        st.session_state.opcion = "Calculadora"
    if ui.button(text="Información", key="btn1", className="text-white w-full"):
        st.session_state.opcion = "Informacion"
   

if st.session_state.opcion == "Informacion":
    st.markdown("""
        <style>
        :root{
            --bg:#0f172a;
            --card:#020617;
            --primary:#38bdf8;
            --secondary:#22c55e;
            --accent:#a78bfa;
            --danger:#f43f5e;
            --warning:#facc15;
            --text:#e5e7eb;
            --muted:#94a3b8;
        }

        .stApp{
            background:var(--bg);
            color:var(--text);
        }

        h2,h3,h4,h5,h6{
            color:#000000 !important;
        }

        p,span,label{
            color:var(--text)!important;
        }

        .stMetric{
            background:var(--card);
            border-radius:12px;
            padding:10px;
        }

        div[data-testid="stFileUploader"],
        div[data-testid="stDataFrame"]{
            background:var(--card);
            border-radius:12px;
            padding:10px;
        }

        button[kind="primary"]{
            background:var(--primary)!important;
            color:#020617!important;
            border-radius:10px!important;
            border:none!important;
        }

        button[kind="secondary"]{
            background:var(--accent)!important;
            color:#020617!important;
            border-radius:10px!important;
            border:none!important;
        }

        .stCheckbox, .stSelectbox, .stSlider{
            background:var(--card);
            border-radius:10px;
            padding:6px;
        }
        </style>
    """, unsafe_allow_html=True)

    tabs = ui.tabs(options=["Usos", "Definicion", "Formulas"], default_value="Usos", key="tabs1")

    if tabs == "Usos":
        with ui.card(key="card_u1"):
            ui.element("span", children=["🏦 Sector Financiero"], className="font-semibold", key="label_u1")
            ui.element("p", children=["Evaluación de riesgo crediticio, detección de fraudes en transacciones, análisis de morosidad y predicción de incumplimientos de pago."], key="text_u1")
        with ui.card(key="card_u2"):
            ui.element("span", children=["🏥 Medicina y Salud"], className="font-semibold", key="label_u2")
            ui.element("p", children=["Diagnóstico de enfermedades, predicción de riesgo de patologías, análisis de supervivencia y clasificación de pacientes según factores de riesgo."], key="text_u2")
        with ui.card(key="card_u3"):
            ui.element("span", children=["📧 Marketing Digital"], className="font-semibold", key="label_u3")
            ui.element("p", children=["Predicción de tasa de clics en anuncios, segmentación de clientes, análisis de conversión y predicción de cancelación de suscripciones."], key="text_u3")
        with ui.card(key="card_u4"):
            ui.element("span", children=["🛡️ Seguros"], className="font-semibold", key="label_u4")
            ui.element("p", children=["Evaluación de riesgo de asegurados, predicción de reclamaciones, detección de fraudes en pólizas y cálculo de primas personalizadas."], key="text_u4")
        with ui.card(key="card_u5"):
            ui.element("span", children=["🎓 Educación"], className="font-semibold", key="label_u5")
            ui.element("p", children=["Predicción de deserción estudiantil, análisis de probabilidad de aprobación, identificación de estudiantes en riesgo y personalización de intervenciones educativas."], key="text_u5")
        with ui.card(key="card_u6"):
            ui.element("span", children=["🏭 Manufactura"], className="font-semibold", key="label_u6")
            ui.element("p", children=["Control de calidad, detección de productos defectuosos, predicción de fallos en maquinaria y optimización de procesos de producción."], key="text_u6")
        with ui.card(key="card_u7"):
            ui.element("span", children=["👥 Recursos Humanos"], className="font-semibold", key="label_u7")
            ui.element("p", children=["Predicción de rotación de empleados, análisis de contratación, evaluación de desempeño y identificación de candidatos con mayor probabilidad de éxito."], key="text_u7")
        with ui.card(key="card_u8"):
            ui.element("span", children=["🌐 Telecomunicaciones"], className="font-semibold", key="label_u8")
            ui.element("p", children=["Predicción de abandono de clientes, análisis de patrones de uso, detección de anomalías en la red y optimización de campañas de retención."], key="text_u8")
        with ui.card(key="card_u9"):
            ui.element("span", children=["🛒 E-commerce"], className="font-semibold", key="label_u9")
            ui.element("p", children=["Predicción de compra, análisis de abandono de carritos, recomendación de productos y segmentación de comportamiento de usuarios."], key="text_u9")
        with ui.card(key="card_u10"):
            ui.element("span", children=["⚖️ Justicia y Seguridad"], className="font-semibold", key="label_u10")
            ui.element("p", children=["Predicción de reincidencia criminal, análisis de riesgo de liberación condicional, clasificación de delitos y apoyo en decisiones judiciales."], key="text_u10")

    elif tabs == "Definicion":
        with ui.card(key="card_d1"):
            ui.element("span", children=["📌 ¿Qué es la Regresión Logística?"], className="font-semibold text-lg", key="label_d1")
            ui.element("p", children=["La regresión logística es un algoritmo de machine learning supervisado utilizado para problemas de clasificación. A pesar de su nombre, no se usa para regresión sino para predecir la probabilidad de que una observación pertenezca a una clase específica."], key="text_d1")
            ui.element("p", children=["Funciona mediante la transformación de una combinación lineal de variables de entrada a través de la función sigmoide, produciendo valores entre 0 y 1 que pueden interpretarse como probabilidades."], key="text_d2")
        with ui.card(key="card_d2"):
            ui.element("span", children=["🎯 Tipos de Regresión Logística"], className="font-semibold text-lg", key="label_d2")
            ui.element("p", children=["Binaria: Clasifica entre dos categorías (Sí/No, 0/1, Verdadero/Falso). Es el tipo más común y predice resultados dicotómicos."], key="text_d3")
            ui.element("p", children=["Multinomial: Maneja tres o más categorías sin orden específico (por ejemplo: tipo de producto A, B o C)."], key="text_d4")
            ui.element("p", children=["Ordinal: Clasifica categorías con un orden natural (por ejemplo: bajo, medio, alto)."], key="text_d5")
        with ui.card(key="card_d3"):
            ui.element("span", children=["⚡ Ventajas Principales"], className="font-semibold text-lg", key="label_d3")
            ui.element("p", children=["Simplicidad e Interpretabilidad: Fácil de implementar y entender. Los coeficientes indican la importancia de cada variable."], key="text_d6")
            ui.element("p", children=["Eficiencia Computacional: Requiere pocos recursos y es rápida en entrenamiento y predicción."], key="text_d7")
            ui.element("p", children=["Probabilidades: Proporciona probabilidades en lugar de solo etiquetas, útil para análisis de riesgo."], key="text_d8")
            ui.element("p", children=["Regularización: Soporta técnicas como L1 y L2 para evitar sobreajuste."], key="text_d9")
        with ui.card(key="card_d4"):
            ui.element("span", children=["🔍 Función Sigmoide"], className="font-semibold text-lg", key="label_d4")
            ui.element("p", children=["El corazón de la regresión logística es la función sigmoide (o logística), que transforma cualquier valor real en un rango entre 0 y 1. Esta característica la hace perfecta para modelar probabilidades."], key="text_d10")
            ui.element("p", children=["La curva en forma de S permite que valores muy negativos se acerquen a 0, valores muy positivos se acerquen a 1, y valores cercanos a 0 produzcan probabilidades intermedias alrededor de 0.5."], key="text_d11")
        with ui.card(key="card_d5"):
            ui.element("span", children=["📊 Comparación con Regresión Lineal"], className="font-semibold text-lg", key="label_d5")
            ui.element("p", children=["Regresión Lineal: Predice valores continuos sin límites (puede dar cualquier número). Usa función lineal directa."], key="text_d12")
            ui.element("p", children=["Regresión Logística: Predice categorías y probabilidades acotadas entre 0 y 1. Usa función sigmoide no lineal."], key="text_d13")
            ui.element("p", children=["La logística es más apropiada cuando la variable objetivo es categórica en lugar de numérica continua."], key="text_d14")
        with ui.card(key="card_d6"):
            ui.element("span", children=["🎲 Umbral de Decisión"], className="font-semibold text-lg", key="label_d6")
            ui.element("p", children=["La regresión logística produce probabilidades, pero para clasificar necesitamos un umbral. El estándar es 0.5: si P(y=1) ≥ 0.5 clasificamos como 1, si no como 0."], key="text_d15")
            ui.element("p", children=["Este umbral puede ajustarse según las necesidades del problema. Por ejemplo, en detección de fraudes podríamos usar 0.3 para ser más sensibles."], key="text_d16")
        with ui.card(key="card_d7"):
            ui.element("span", children=["💡 Cuándo Usar Regresión Logística"], className="font-semibold text-lg", key="label_d7")
            ui.element("p", children=["Variable objetivo binaria o categórica, relación aproximadamente lineal entre variables independientes y log-odds, necesidad de interpretabilidad de resultados, conjunto de datos con muchas observaciones, cuando se requiere entrenamiento rápido y predicciones eficientes."], key="text_d17")

    elif tabs == "Formulas":
        with ui.card(key="card_f1"):
            ui.element("span", children=["📐 Función Sigmoide"], className="font-semibold text-lg", key="label_f1")
            st.latex(r"\sigma(z) = \frac{1}{1 + e^{-z}}")
        with ui.card(key="card_f2"):
            ui.element("span", children=["🎯 Modelo de Regresión Logística"], className="font-semibold text-lg", key="label_f2")
            st.latex(r"z = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n")
            st.latex(r"P(y=1|x) = \frac{1}{1 + e^{-z}}")
        with ui.card(key="card_f3"):
            ui.element("span", children=["📊 Función de Pérdida (Log Loss)"], className="font-semibold text-lg", key="label_f3")
            st.latex(r"J(\beta) = -\frac{1}{m}\sum_{i=1}^{m}[y_i\log(h(x_i)) + (1-y_i)\log(1-h(x_i))]")
        with ui.card(key="card_f4"):
            ui.element("span", children=["🔄 Gradiente Descendente"], className="font-semibold text-lg", key="label_f4")
            st.latex(r"\beta_j := \beta_j - \alpha\frac{\partial J(\beta)}{\partial \beta_j}")
        with ui.card(key="card_f5"):
            ui.element("span", children=["🎲 Odds y Log-Odds"], className="font-semibold text-lg", key="label_f5")
            st.latex(r"Odds = \frac{P(y=1)}{P(y=0)} = \frac{P(y=1)}{1-P(y=1)}")
            st.latex(r"Log(Odds) = \log\left(\frac{P(y=1)}{1-P(y=1)}\right) = z")
        with ui.card(key="card_f6"):
            ui.element("span", children=["✅ Regla de Clasificación"], className="font-semibold text-lg", key="label_f6")
            st.latex(r"\hat{y} = \begin{cases} 1 & \text{si } P(y=1|x) \geq 0.5 \\ 0 & \text{si } P(y=1|x) < 0.5 \end{cases}")




















elif st.session_state.opcion == "Ejemplos":
    st.markdown("""
        <style>
        :root{
            --bg:#0f172a;
            --card:#020617;
            --primary:#38bdf8;
            --secondary:#22c55e;
            --accent:#a78bfa;
            --danger:#f43f5e;
            --warning:#facc15;
            --text:#e5e7eb;
            --muted:#94a3b8;
        }

        .stApp{
            background:var(--bg);
            color:var(--text);
        }

        h1,h2,h3,h4,h5,h6,p,span,label{
            color:var(--text)!important;
        }

        .stMetric{
            background:var(--card);
            border-radius:12px;
            padding:10px;
        }

        div[data-testid="stFileUploader"],
        div[data-testid="stDataFrame"]{
            background:var(--card);
            border-radius:12px;
            padding:10px;
        }

        button[kind="primary"]{
            background:var(--primary)!important;
            color:#020617!important;
            border-radius:10px!important;
            border:none!important;
        }

        button[kind="secondary"]{
            background:var(--accent)!important;
            color:#020617!important;
            border-radius:10px!important;
            border:none!important;
        }

        .stCheckbox, .stSelectbox, .stSlider{
            background:var(--card);
            border-radius:10px;
            padding:6px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🏎️ Predicción de Tipo de Carrera de Fórmula 1")
    st.markdown("### Ejemplo práctico: Predecir si una carrera es tipo 'Race' o no")

    with st.spinner("Cargando datos de la API..."):
        try:
            df_raw = Getdata.get_dataset()
            st.success("Datos cargados exitosamente")

            with ui.card(key="card_step1"):
                st.markdown("#### Exploración del Dataset")
                st.dataframe(df_raw.head(10), use_container_width=True)
                st.metric("Total de carreras", len(df_raw))
                st.markdown("Tipos de carreras")
                st.dataframe(df_raw["type"].value_counts().reset_index(), use_container_width=True)

            with ui.card(key="card_step2"):
                df_clean = Getdata.cleanData(df_raw)
                st.dataframe(df_clean.head(10), use_container_width=True)

            with ui.card(key="card_step3"):
                df_model = df_clean.copy()
                df_model["is_race"] = (df_model["type"] == "Race").astype(int)

                c1,c2,c3 = st.columns(3)
                with c1:
                    st.metric("Race", df_model["is_race"].sum())
                with c2:
                    st.metric("No Race", len(df_model) - df_model["is_race"].sum())
                with c3:
                    st.metric("% Race", f"{df_model['is_race'].mean()*100:.1f}%")

            with ui.card(key="card_step4"):
                le_circuit = LabelEncoder()
                le_country = LabelEncoder()
                le_status = LabelEncoder()

                df_model["circuit_encoded"] = le_circuit.fit_transform(df_model["circuit"])
                df_model["country_encoded"] = le_country.fit_transform(df_model["country"])
                df_model["status_encoded"] = le_status.fit_transform(df_model["status"])
                df_model["laps_total_filled"] = df_model["laps_total"].fillna(0)
                df_model["month"] = pd.to_datetime(df_model["date"]).dt.month

                features = ["circuit_encoded","country_encoded","status_encoded","laps_total_filled","month"]
                st.dataframe(df_model[features + ["is_race"]].head(), use_container_width=True)

            with ui.card(key="card_step5"):
                X = df_model[features]
                y = df_model["is_race"]
                X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

                c1,c2 = st.columns(2)
                with c1:
                    st.metric("Entrenamiento", len(X_train))
                with c2:
                    st.metric("Prueba", len(X_test))

            with ui.card(key="card_step6"):
                model = LogisticRegression(max_iter=1000, random_state=42)
                model.fit(X_train, y_train)
                st.success("Modelo entrenado")

                coef_df = pd.DataFrame({
                    "Característica": features,
                    "Coeficiente": model.coef_[0]
                }).sort_values("Coeficiente", key=np.abs, ascending=False)

                st.dataframe(coef_df, use_container_width=True)
                st.metric("Intercepto", f"{model.intercept_[0]:.4f}")

            with ui.card(key="card_step7"):
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:,1]

                c1,c2,c3,c4 = st.columns(4)
                with c1:
                    st.metric("Accuracy", f"{accuracy_score(y_test,y_pred)*100:.2f}%")
                with c2:
                    st.metric("Precision", f"{precision_score(y_test,y_pred)*100:.2f}%")
                with c3:
                    st.metric("Recall", f"{recall_score(y_test,y_pred)*100:.2f}%")
                with c4:
                    st.metric("F1", f"{f1_score(y_test,y_pred)*100:.2f}%")

                st.dataframe(pd.DataFrame(confusion_matrix(y_test,y_pred),
                    columns=["Pred No Race","Pred Race"],
                    index=["Real No Race","Real Race"]), use_container_width=True)

            with ui.card(key="card_step8"):
                ejemplo = X_test.iloc[[0]]
                real = y_test.iloc[0]

                pred = model.predict(ejemplo)[0]
                prob = model.predict_proba(ejemplo)[0]

                c1,c2,c3 = st.columns(3)
                with c1:
                    st.metric("Predicción", "Race" if pred==1 else "No Race")
                with c2:
                    st.metric("Prob Race", f"{prob[1]*100:.2f}%")
                with c3:
                    st.metric("Prob No Race", f"{prob[0]*100:.2f}%")

                z = model.intercept_[0] + np.sum(model.coef_[0] * ejemplo.values[0])
                prob_manual = 1 / (1 + np.exp(-z))
                st.latex(f"P(Race)=\\frac{{1}}{{1+e^{{-{z:.4f}}}}}={prob_manual:.4f}")

                if pred == real:
                    st.success("Predicción correcta")
                else:
                    st.error("Predicción incorrecta")

        except Exception as e:
            st.error(str(e))
























elif st.session_state.opcion == "Calculadora":
    st.markdown("""
        <style>
        :root{
            --bg:#0f172a;
            --card:#020617;
            --primary:#38bdf8;
            --secondary:#22c55e;
            --accent:#a78bfa;
            --danger:#f43f5e;
            --warning:#facc15;
            --text:#e5e7eb;
            --muted:#94a3b8;
        }

        .stApp{
            background:var(--bg);
            color:var(--text);
        }

        h1,h2,h3,h4,h5,h6,p,span,label{
            color:var(--text)!important;
        }

        .stMetric{
            background:var(--card);
            border-radius:12px;
            padding:10px;
        }

        div[data-testid="stFileUploader"],
        div[data-testid="stDataFrame"]{
            background:var(--card);
            border-radius:12px;
            padding:10px;
        }

        button[kind="primary"]{
            background:var(--primary)!important;
            color:#020617!important;
            border-radius:10px!important;
            border:none!important;
        }

        button[kind="secondary"]{
            background:var(--accent)!important;
            color:#020617!important;
            border-radius:10px!important;
            border:none!important;
        }

        .stCheckbox, .stSelectbox, .stSlider, .stNumberInput{
            background:var(--card);
            border-radius:10px;
            padding:6px;
        }

        .explicacion-box{
            background:var(--card);
            border-left:4px solid var(--primary);
            padding:20px;
            border-radius:12px;
            margin:20px 0;
        }

        .prediccion-box{
            background:linear-gradient(135deg, rgba(56,189,248,0.2), rgba(167,139,250,0.2));
            border:2px solid var(--primary);
            padding:20px;
            border-radius:12px;
            margin:15px 0;
        }

        .resultado-positivo{
            background:linear-gradient(135deg, rgba(34,197,94,0.3), rgba(34,197,94,0.1));
            border-left:5px solid var(--secondary);
            padding:15px;
            border-radius:8px;
        }

        .resultado-negativo{
            background:linear-gradient(135deg, rgba(244,63,94,0.3), rgba(244,63,94,0.1));
            border-left:5px solid var(--danger);
            padding:15px;
            border-radius:8px;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("🧮 Calculadora de Regresión Logística")
    st.markdown("### Carga tu dataset y entrena tu modelo de clasificación")

    uploaded_file = st.file_uploader("📁 Sube tu archivo CSV o Excel", type=['csv', 'xlsx', 'xls'])

    if uploaded_file is not None:
        try:
            if uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)
            else:
                df = pd.read_excel(uploaded_file)

            with ui.card(key="calc_card1"):
                st.markdown("#### 📊 Vista previa del Dataset")
                st.dataframe(df.head(10), use_container_width=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Filas", df.shape[0])
                with c2:
                    st.metric("Columnas", df.shape[1])
                with c3:
                    st.metric("Valores Nulos", df.isnull().sum().sum())

            with ui.card(key="calc_card_clean"):
                st.markdown("#### 🧹 Limpieza de Datos")

                c1, c2, c3 = st.columns(3)
                with c1:
                    remove_duplicates = st.checkbox("Eliminar filas duplicadas", True)
                with c2:
                    handle_nulls = st.selectbox("Manejar valores nulos", ["Eliminar filas con nulos", "Rellenar con media/moda", "No hacer nada"])
                with c3:
                    remove_outliers = st.checkbox("Eliminar outliers", False)

                if st.button("Aplicar Limpieza", use_container_width=True):
                    original = df.shape[0]

                    if remove_duplicates:
                        df = df.drop_duplicates()

                    if handle_nulls == "Eliminar filas con nulos":
                        df = df.dropna()
                    elif handle_nulls == "Rellenar con media/moda":
                        for c in df.columns:
                            if df[c].dtype in ["int64","float64"]:
                                df[c] = df[c].fillna(df[c].mean())
                            else:
                                df[c] = df[c].fillna(df[c].mode()[0])

                    if remove_outliers:
                        for c in df.select_dtypes(include=["int64","float64"]).columns:
                            q1 = df[c].quantile(0.25)
                            q3 = df[c].quantile(0.75)
                            iqr = q3 - q1
                            df = df[(df[c] >= q1 - 1.5*iqr) & (df[c] <= q3 + 1.5*iqr)]

                    st.success(f"Dataset limpio: {original} → {df.shape[0]} filas")
                    st.dataframe(df.head(), use_container_width=True)

            with ui.card(key="calc_card2"):
                st.markdown("#### ⚙️ Configuración del Modelo")

                binary_cols = [c for c in df.columns if df[c].nunique() == 2]

                target_column = st.selectbox("Variable objetivo", df.columns, index=df.columns.tolist().index(binary_cols[0]) if binary_cols else 0)
                feature_columns = st.multiselect("Características", [c for c in df.columns if c != target_column])

                test_size = st.slider("Tamaño prueba (%)", 10, 50, 20)
                normalize = st.checkbox("Normalizar datos", True)

            if target_column and feature_columns:
                if st.button("Entrenar Modelo", use_container_width=True):
                    df_model = df.dropna(subset=[target_column] + feature_columns)
                    X = df_model[feature_columns]
                    y = df_model[target_column]

                    encoders = {}
                    for c in X.columns:
                        if X[c].dtype == "object":
                            le = LabelEncoder()
                            X[c] = le.fit_transform(X[c].astype(str))
                            encoders[c] = le

                    target_labels = {}
                    if y.dtype == "object":
                        target_labels = {0: str(y.unique()[0]), 1: str(y.unique()[1])}
                        y = y.map({y.unique()[0]:0, y.unique()[1]:1}).astype(int)
                    else:
                        y = y.astype(int)
                        target_labels = {0: "Clase 0", 1: "Clase 1"}

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size/100, random_state=42, stratify=y)

                    scaler = None
                    if normalize:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                    model = LogisticRegression(max_iter=1000)
                    model.fit(X_train, y_train)

                    st.success("✅ Modelo entrenado correctamente")

                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test)[:,1]
                    y_pred_train = model.predict(X_train)

                    accuracy_test = accuracy_score(y_test, y_pred)
                    accuracy_train = accuracy_score(y_train, y_pred_train)
                    precision = precision_score(y_test, y_pred, zero_division=0)
                    recall = recall_score(y_test, y_pred, zero_division=0)
                    f1 = f1_score(y_test, y_pred, zero_division=0)
                    roc_auc = roc_auc_score(y_test, y_prob)

                    with ui.card(key="metricas_principales"):
                        st.markdown("#### 📈 Métricas del Modelo")
                        col1, col2, col3, col4, col5 = st.columns(5)
                        with col1:
                            st.metric("Accuracy", f"{accuracy_test*100:.2f}%")
                        with col2:
                            st.metric("Precision", f"{precision*100:.2f}%")
                        with col3:
                            st.metric("Recall", f"{recall*100:.2f}%")
                        with col4:
                            st.metric("F1-Score", f"{f1:.4f}")
                        with col5:
                            st.metric("ROC-AUC", f"{roc_auc:.4f}")

                    with ui.card(key="curva_sigmoidea"):
                        st.markdown("#### 📈 Curva Sigmoidea (Función Logística)")
                        
                        if len(feature_columns) == 1:
                            X_range = np.linspace(X[feature_columns[0]].min(), X[feature_columns[0]].max(), 300).reshape(-1, 1)
                            if scaler:
                                X_range_scaled = scaler.transform(X_range)
                            else:
                                X_range_scaled = X_range
                            y_prob_range = model.predict_proba(X_range_scaled)[:,1]
                            
                            fig_sigmoid = go.Figure()
                            
                            fig_sigmoid.add_trace(go.Scatter(
                                x=X_range.flatten(),
                                y=y_prob_range,
                                mode='lines',
                                name='Curva Sigmoidea',
                                line=dict(color='#38bdf8', width=4)
                            ))
                            
                            X_test_original = X_test if not normalize else scaler.inverse_transform(X_test)
                            fig_sigmoid.add_trace(go.Scatter(
                                x=X_test_original.flatten(),
                                y=y_test,
                                mode='markers',
                                name='Datos Reales',
                                marker=dict(
                                    size=10,
                                    color=['#22c55e' if val == 1 else '#f43f5e' for val in y_test],
                                    line=dict(width=2, color='white')
                                )
                            ))
                            
                            fig_sigmoid.add_hline(y=0.5, line_dash="dash", line_color="yellow", 
                                                 annotation_text="Umbral de decisión (0.5)")
                            
                            fig_sigmoid.update_layout(
                                template='plotly_dark',
                                height=500,
                                title='Función Sigmoidea: σ(z) = 1 / (1 + e^(-z))',
                                xaxis_title=feature_columns[0],
                                yaxis_title='Probabilidad P(Y=1)',
                                hovermode='closest',
                                showlegend=True
                            )
                            
                            st.plotly_chart(fig_sigmoid, use_container_width=True)
                            
                            st.markdown("""
                            <div class="explicacion-box">
                            <h4>🔵 ¿Qué es la Curva Sigmoidea?</h4>
                            <ul>
                                <li>Es la función que transforma cualquier valor en una probabilidad entre 0 y 1</li>
                                <li><b>Forma de S característica:</b> valores bajos → probabilidad cercana a 0, valores altos → probabilidad cercana a 1</li>
                                <li><b>Punto de inflexión:</b> cuando la probabilidad es 0.5 (umbral de decisión)</li>
                                <li>Los puntos verdes son casos positivos reales, los rojos son negativos</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            z_vals = np.linspace(-10, 10, 300)
                            sigmoid_vals = 1 / (1 + np.exp(-z_vals))
                            
                            fig_sigmoid = go.Figure()
                            
                            fig_sigmoid.add_trace(go.Scatter(
                                x=z_vals,
                                y=sigmoid_vals,
                                mode='lines',
                                name='σ(z) = 1/(1+e^(-z))',
                                line=dict(color='#38bdf8', width=4),
                                fill='tozeroy',
                                fillcolor='rgba(56, 189, 248, 0.2)'
                            ))
                            
                            fig_sigmoid.add_hline(y=0.5, line_dash="dash", line_color="yellow", 
                                                 annotation_text="Umbral 0.5")
                            fig_sigmoid.add_vline(x=0, line_dash="dash", line_color="white", 
                                                 annotation_text="z = 0")
                            
                            fig_sigmoid.add_trace(go.Scatter(
                                x=[-10, 0, 10],
                                y=[0, 0.5, 1],
                                mode='markers',
                                name='Puntos clave',
                                marker=dict(size=12, color='#22c55e', symbol='circle')
                            ))
                            
                            fig_sigmoid.update_layout(
                                template='plotly_dark',
                                height=500,
                                title='Función Sigmoidea: σ(z) = 1 / (1 + e^(-z))',
                                xaxis_title='z = β₀ + β₁x₁ + β₂x₂ + ... + βₙxₙ',
                                yaxis_title='Probabilidad P(Y=1)',
                                hovermode='closest'
                            )
                            
                            st.plotly_chart(fig_sigmoid, use_container_width=True)
                            
                            st.markdown("""
                            <div class="explicacion-box">
                            <h4>🔵 ¿Qué es la Curva Sigmoidea?</h4>
                            <ul>
                                <li><b>Función logística:</b> σ(z) = 1 / (1 + e^(-z))</li>
                                <li>Convierte el resultado lineal (z) en una probabilidad entre 0 y 1</li>
                                <li><b>Cuando z = 0:</b> probabilidad = 0.5 (punto de decisión)</li>
                                <li><b>z muy negativo:</b> probabilidad → 0 (clase negativa)</li>
                                <li><b>z muy positivo:</b> probabilidad → 1 (clase positiva)</li>
                                <li>La pendiente en el centro muestra qué tan "seguras" son las predicciones</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)

                    with ui.card(key="matriz_confusion"):
                        st.markdown("#### 🎯 Matriz de Confusión")
                        
                        cm = confusion_matrix(y_test, y_pred)
                        
                        fig_cm = go.Figure(data=go.Heatmap(
                            z=cm,
                            x=[f'Predicho: {target_labels[0]}', f'Predicho: {target_labels[1]}'],
                            y=[f'Real: {target_labels[0]}', f'Real: {target_labels[1]}'],
                            text=cm,
                            texttemplate='%{text}',
                            textfont={"size": 20},
                            colorscale='Viridis',
                            showscale=True
                        ))
                        
                        fig_cm.update_layout(
                            template='plotly_dark',
                            height=400,
                            title='Matriz de Confusión',
                            xaxis_title='Predicción',
                            yaxis_title='Valor Real'
                        )
                        
                        st.plotly_chart(fig_cm, use_container_width=True)

                        st.markdown(f"""
                        <div class="explicacion-box">
                        <h4>🎯 Lectura de la Matriz:</h4>
                        <ul>
                            <li><b>Verdaderos Negativos (TN):</b> {cm[0,0]} - {target_labels[0]} correctamente clasificados</li>
                            <li><b>Falsos Positivos (FP):</b> {cm[0,1]} - {target_labels[0]} clasificados como {target_labels[1]}</li>
                            <li><b>Falsos Negativos (FN):</b> {cm[1,0]} - {target_labels[1]} clasificados como {target_labels[0]}</li>
                            <li><b>Verdaderos Positivos (TP):</b> {cm[1,1]} - {target_labels[1]} correctamente clasificados</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    with ui.card(key="curva_roc"):
                        st.markdown("#### 📈 Curva ROC")
                        
                        fpr, tpr, thresholds = roc_curve(y_test, y_prob)
                        
                        fig_roc = go.Figure()
                        
                        fig_roc.add_trace(go.Scatter(
                            x=fpr,
                            y=tpr,
                            mode='lines',
                            name=f'ROC (AUC = {roc_auc:.4f})',
                            line=dict(color='#38bdf8', width=3),
                            fill='tozeroy',
                            fillcolor='rgba(56, 189, 248, 0.2)'
                        ))
                        
                        fig_roc.add_trace(go.Scatter(
                            x=[0, 1],
                            y=[0, 1],
                            mode='lines',
                            name='Clasificador Aleatorio',
                            line=dict(color='#f43f5e', width=2, dash='dash')
                        ))
                        
                        fig_roc.update_layout(
                            template='plotly_dark',
                            height=500,
                            title=f'Curva ROC - AUC = {roc_auc:.4f}',
                            xaxis_title='Tasa de Falsos Positivos (FPR)',
                            yaxis_title='Tasa de Verdaderos Positivos (TPR)',
                            hovermode='closest',
                            showlegend=True
                        )
                        
                        st.plotly_chart(fig_roc, use_container_width=True)

                        st.markdown(f"""
                        <div class="explicacion-box">
                        <h4>📊 Interpretación ROC-AUC = {roc_auc:.4f}</h4>
                        <ul>
                            <li>Mide la capacidad del modelo para distinguir entre clases</li>
                            <li><b>1.0 = Perfecto:</b> Separa completamente las clases</li>
                            <li><b>0.5 = Aleatorio:</b> No mejor que lanzar una moneda</li>
                            <li><b>Tu modelo:</b> {'🟢 Excelente discriminación' if roc_auc > 0.9 else '🟡 Buena discriminación' if roc_auc > 0.8 else '🟠 Discriminación aceptable' if roc_auc > 0.7 else '🔴 Discriminación pobre'}</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    with ui.card(key="predictor"):
                        st.markdown("#### 🔮 Predictor Interactivo")
                        st.markdown("##### Ingresa valores para realizar una predicción")
                        
                        input_data = {}
                        cols = st.columns(min(3, len(feature_columns)))
                        
                        for idx, feat in enumerate(feature_columns):
                            with cols[idx % 3]:
                                if feat in encoders:
                                    options = encoders[feat].classes_.tolist()
                                    input_data[feat] = st.selectbox(f"{feat}", options, key=f"pred_{feat}")
                                else:
                                    min_val = float(df[feat].min())
                                    max_val = float(df[feat].max())
                                    mean_val = float(df[feat].mean())
                                    input_data[feat] = st.number_input(
                                        f"{feat}",
                                        min_value=min_val,
                                        max_value=max_val,
                                        value=mean_val,
                                        key=f"pred_{feat}"
                                    )
                        
                        if st.button("🎯 Realizar Predicción", use_container_width=True, type="primary"):
                            input_df = pd.DataFrame([input_data])
                            
                            for col in input_df.columns:
                                if col in encoders:
                                    input_df[col] = encoders[col].transform(input_df[col].astype(str))
                            
                            if scaler:
                                input_scaled = scaler.transform(input_df)
                            else:
                                input_scaled = input_df.values
                            
                            pred_class = model.predict(input_scaled)[0]
                            pred_proba = model.predict_proba(input_scaled)[0]
                            
                            pred_label = target_labels[pred_class]
                            prob_class_0 = pred_proba[0] * 100
                            prob_class_1 = pred_proba[1] * 100
                            
                            resultado_class = "resultado-positivo" if pred_class == 1 else "resultado-negativo"
                            
                            st.markdown(f"""
                            <div class="{resultado_class}">
                            <h3>🎯 Resultado de la Predicción</h3>
                            <h2 style="font-size:2em; margin:10px 0;">Clase Predicha: {pred_label}</h2>
                            </div>
                            """, unsafe_allow_html=True)
                            
                            col1, col2 = st.columns(2)
                            with col1:
                                st.metric(
                                    f"Probabilidad {target_labels[0]}",
                                    f"{prob_class_0:.2f}%",
                                    delta=None
                                )
                            with col2:
                                st.metric(
                                    f"Probabilidad {target_labels[1]}",
                                    f"{prob_class_1:.2f}%",
                                    delta=None
                                )
                            
                            fig_pred = go.Figure()
                            
                            fig_pred.add_trace(go.Bar(
                                x=[target_labels[0], target_labels[1]],
                                y=[prob_class_0, prob_class_1],
                                marker_color=['#f43f5e', '#22c55e'],
                                text=[f"{prob_class_0:.2f}%", f"{prob_class_1:.2f}%"],
                                textposition='auto',
                                textfont=dict(size=16, color='white')
                            ))
                            
                            fig_pred.update_layout(
                                template='plotly_dark',
                                height=400,
                                title='Distribución de Probabilidades',
                                yaxis_title='Probabilidad (%)',
                                xaxis_title='Clase',
                                showlegend=False
                            )
                            
                            st.plotly_chart(fig_pred, use_container_width=True)
                            
                            st.markdown(f"""
                            <div class="prediccion-box">
                            <h4>📋 Detalles de la Predicción</h4>
                            <p><b>Valores de entrada:</b></p>
                            <ul>
                                {''.join([f'<li><b>{k}:</b> {v}</li>' for k, v in input_data.items()])}
                            </ul>
                            <p><b>Resultado del modelo:</b></p>
                            <ul>
                                <li><b>Clase predicha:</b> {pred_label} (valor {pred_class})</li>
                                <li><b>Confianza:</b> {max(prob_class_0, prob_class_1):.2f}%</li>
                                <li><b>Interpretación:</b> {'🟢 Alta confianza' if max(prob_class_0, prob_class_1) > 80 else '🟡 Confianza moderada' if max(prob_class_0, prob_class_1) > 60 else '🟠 Baja confianza'}</li>
                            </ul>
                            <p><b>¿Cómo funciona?</b></p>
                            <ul>
                                <li>El modelo calcula: z = β₀ + β₁×x₁ + β₂×x₂ + ...</li>
                                <li>Luego aplica la función sigmoidea: P(Y=1) = 1 / (1 + e^(-z))</li>
                                <li>Si P(Y=1) > 0.5 → Predice clase 1, sino → Predice clase 0</li>
                                <li>Las probabilidades siempre suman 100%</li>
                            </ul>
                            </div>
                            """, unsafe_allow_html=True)

                    with ui.card(key="explicacion_metricas"):
                        st.markdown("#### 📚 Interpretación de Resultados")
                        
                        st.markdown(f"""
                        <div class="explicacion-box">
                        <h4>📊 Análisis de Métricas:</h4>
                        
                        <p><b>🎯 Accuracy = {accuracy_test*100:.2f}%</b></p>
                        <ul>
                            <li>Porcentaje de predicciones correctas totales</li>
                            <li><b>Tu modelo acierta en {accuracy_test*100:.2f}% de los casos</b></li>
                            <li>{'🟢 Excelente' if accuracy_test > 0.9 else '🟡 Bueno' if accuracy_test > 0.8 else '🟠 Aceptable' if accuracy_test > 0.7 else '🔴 Necesita mejorar'}</li>
                        </ul>
                        
                        <p><b>🎯 Precision = {precision*100:.2f}%</b></p>
                        <ul>
                            <li>De las predicciones positivas, ¿cuántas son correctas?</li>
                            <li><b>{precision*100:.2f}% de las predicciones {target_labels[1]} son correctas</b></li>
                            <li>Importante cuando los falsos positivos son costosos</li>
                        </ul>
                        
                        <p><b>🎯 Recall = {recall*100:.2f}%</b></p>
                        <ul>
                            <li>De todos los casos {target_labels[1]} reales, ¿cuántos detectó?</li>
                            <li><b>El modelo detecta {recall*100:.2f}% de los casos {target_labels[1]}</b></li>
                            <li>Importante cuando los falsos negativos son costosos</li>
                        </ul>
                        
                        <p><b>🎯 F1-Score = {f1:.4f}</b></p>
                        <ul>
                            <li>Media armónica entre Precision y Recall</li>
                            <li>Útil cuando necesitas balance entre ambas métricas</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="explicacion-box">
                        <h4>🔍 Análisis de Generalización:</h4>
                        <ul>
                            <li>Accuracy Train: {accuracy_train*100:.2f}%</li>
                            <li>Accuracy Test: {accuracy_test*100:.2f}%</li>
                            <li>Diferencia: {abs(accuracy_train - accuracy_test)*100:.2f}%</li>
                            <li>{'🟢 Modelo bien generalizado' if abs(accuracy_train - accuracy_test) < 0.05 else '🟡 Ligero sobreajuste' if abs(accuracy_train - accuracy_test) < 0.1 else '🔴 Sobreajuste significativo'}</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.markdown(f"""
                        <div class="explicacion-box">
                        <h4>💡 Recomendaciones:</h4>
                        <ul>
                            <li>{'✅ Excelente rendimiento general' if accuracy_test > 0.9 else '✅ Buen rendimiento' if accuracy_test > 0.8 else '⚠️ Considera mejorar el modelo'}</li>
                            <li>{'✅ Excelente discriminación entre clases' if roc_auc > 0.9 else '✅ Buena discriminación' if roc_auc > 0.8 else '⚠️ Mejora necesaria en separación de clases'}</li>
                            <li>{'✅ Balance adecuado Precision-Recall' if abs(precision - recall) < 0.1 else '⚠️ Ajusta el umbral según tu caso de uso'}</li>
                            <li>{'✅ Sin sobreajuste significativo' if abs(accuracy_train - accuracy_test) < 0.1 else '⚠️ Considera regularización o más datos'}</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    with ui.card(key="ecuacion_modelo"):
                        st.markdown("#### 🧮 Ecuación del Modelo Logístico")
                        
                        ecuacion = f"z = {model.intercept_[0]:.4f}"
                        for i, feat in enumerate(feature_columns):
                            coef = model.coef_[0][i]
                            signo = "+" if coef >= 0 else ""
                            ecuacion += f" {signo} {coef:.4f}×{feat}"
                        
                        st.code(ecuacion, language="text")
                        st.code("P(Y=1) = 1 / (1 + e^(-z))", language="text")
                        
                        st.markdown("""
                        <div class="explicacion-box">
                        <p><b>Interpretación de Coeficientes:</b></p>
                        <ul>
                            <li><b>Coeficiente positivo:</b> Aumenta la probabilidad de clase 1 cuando la característica aumenta</li>
                            <li><b>Coeficiente negativo:</b> Disminuye la probabilidad de clase 1 cuando la característica aumenta</li>
                            <li><b>Magnitud del coeficiente:</b> Indica qué tan fuerte es el efecto de esa característica</li>
                            <li><b>Intercepto (β₀):</b> Es el valor base cuando todas las características son 0</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)

                        coef_df = pd.DataFrame({
                            "Característica": feature_columns,
                            "Coeficiente": model.coef_[0],
                            "Efecto": ["↑ Aumenta P(Y=1)" if c > 0 else "↓ Disminuye P(Y=1)" for c in model.coef_[0]]
                        }).sort_values("Coeficiente", key=np.abs, ascending=False)

                        fig_coef = px.bar(
                            coef_df,
                            x="Coeficiente",
                            y="Característica",
                            orientation="h",
                            color="Coeficiente",
                            color_continuous_scale=["#f43f5e","#94a3b8","#22c55e"],
                            template="plotly_dark",
                            title="Importancia y Dirección de Características"
                        )
                        fig_coef.update_layout(height=400)
                        st.plotly_chart(fig_coef, use_container_width=True)

                        st.dataframe(coef_df, use_container_width=True, hide_index=True)

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
            import traceback
            st.code(traceback.format_exc())
    else:
        st.info("📁 Carga un archivo CSV o Excel para comenzar el análisis")