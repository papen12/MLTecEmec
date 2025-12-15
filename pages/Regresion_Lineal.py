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




st.sidebar.title("Regresión Lineal")

if "opcion" not in st.session_state:
    st.session_state.opcion = "Informacion"

with st.sidebar:
    if ui.button(text="Calculadora", key="btn6", className=" text-white w-full"):
        st.session_state.opcion = "Calculadora Regresion Linear"
    if ui.button(text="Ejemplos", key="btn5", className=" text-white w-full"):
        st.session_state.opcion = "Ejemplos_Regresion_Lineal"
    if ui.button(text="Información", key="btn4", className=" text-white w-full"):
        st.session_state.opcion = "Informacion Regresion Linear"
    
    
        
        
if st.session_state.opcion == "Informacion Regresion Linear":
    tabs = ui.tabs(options=["Usos", "Definicion", "Formulas"], default_value="Usos", key="tabs1")
    
    if tabs == "Usos":
        with ui.card(key="card_u1"):
            ui.element("span", children=["📈 Sector Financiero"], className="text-gray-700 font-semibold", key="label_u1")
            ui.element("p", children=["Predicción de precios de acciones, valoración de activos, análisis de tendencias del mercado, proyección de ingresos y estimación de retornos de inversión."], key="text_u1")

        with ui.card(key="card_u2"):
            ui.element("span", children=["🏠 Bienes Raíces"], className="text-gray-700 font-semibold", key="label_u2")
            ui.element("p", children=["Estimación de precios de propiedades, valoración inmobiliaria, análisis de factores que afectan el valor de viviendas y predicción de tendencias del mercado inmobiliario."], key="text_u2")

        with ui.card(key="card_u3"):
            ui.element("span", children=["💼 Economía y Negocios"], className="text-gray-700 font-semibold", key="label_u3")
            ui.element("p", children=["Proyección de ventas, análisis de demanda, predicción de crecimiento económico, estimación de costos y planificación presupuestaria."], key="text_u3")

        with ui.card(key="card_u4"):
            ui.element("span", children=["🔬 Ciencia e Investigación"], className="text-gray-700 font-semibold", key="label_u4")
            ui.element("p", children=["Modelado de fenómenos naturales, análisis de relaciones entre variables, predicción de resultados experimentales y estudios de correlación."], key="text_u4")

        with ui.card(key="card_u5"):
            ui.element("span", children=["🏭 Manufactura"], className="text-gray-700 font-semibold", key="label_u5")
            ui.element("p", children=["Optimización de procesos de producción, predicción de tiempos de fabricación, estimación de costos de producción y control de calidad mediante análisis de variables."], key="text_u5")

        with ui.card(key="card_u6"):
            ui.element("span", children=["📊 Marketing y Publicidad"], className="text-gray-700 font-semibold", key="label_u6")
            ui.element("p", children=["Predicción de ROI de campañas, análisis de impacto de inversión publicitaria en ventas, estimación de alcance y optimización de presupuestos de marketing."], key="text_u6")

        with ui.card(key="card_u7"):
            ui.element("span", children=["🌡️ Clima y Medio Ambiente"], className="text-gray-700 font-semibold", key="label_u7")
            ui.element("p", children=["Predicción de temperaturas, análisis de patrones climáticos, modelado de contaminación, estimación de consumo energético y pronósticos meteorológicos."], key="text_u7")

        with ui.card(key="card_u8"):
            ui.element("span", children=["🏥 Salud y Medicina"], className="text-gray-700 font-semibold", key="label_u8")
            ui.element("p", children=["Predicción de dosis de medicamentos, análisis de correlación entre variables clínicas, estimación de tiempos de recuperación y modelado de respuestas a tratamientos."], key="text_u8")

        with ui.card(key="card_u9"):
            ui.element("span", children=["🚗 Transporte y Logística"], className="text-gray-700 font-semibold", key="label_u9")
            ui.element("p", children=["Predicción de tiempos de entrega, estimación de costos de transporte, optimización de rutas basada en variables y análisis de consumo de combustible."], key="text_u9")

        with ui.card(key="card_u10"):
            ui.element("span", children=["📚 Educación"], className="text-gray-700 font-semibold", key="label_u10")
            ui.element("p", children=["Predicción de calificaciones, análisis de factores que influyen en el rendimiento académico, estimación de tasas de graduación y evaluación de programas educativos."], key="text_u10")
    
    elif tabs == "Definicion":
        with ui.card(key="card_d1"):
            ui.element("span", children=["📌 ¿Qué es la Regresión Lineal?"], className="text-gray-700 font-semibold text-lg", key="label_d1")
            ui.element("p", children=["La regresión lineal es un algoritmo de machine learning supervisado utilizado para predecir valores continuos. Modela la relación entre una variable dependiente y una o más variables independientes mediante una ecuación lineal."], key="text_d1")
            ui.element("p", children=["El objetivo es encontrar la mejor línea recta que se ajuste a los datos, minimizando la diferencia entre los valores predichos y los valores reales. Esta línea representa la tendencia general de los datos."], key="text_d2")

        with ui.card(key="card_d2"):
            ui.element("span", children=["🎯 Tipos de Regresión Lineal"], className="text-gray-700 font-semibold text-lg", key="label_d2")
            ui.element("p", children=["Simple: Utiliza una sola variable independiente para predecir la variable dependiente (y = mx + b). Es fácil de visualizar e interpretar."], key="text_d3")
            ui.element("p", children=["Múltiple: Usa dos o más variables independientes para hacer predicciones más complejas y precisas (y = b₀ + b₁x₁ + b₂x₂ + ... + bₙxₙ)."], key="text_d4")
            ui.element("p", children=["Polinomial: Aunque no es estrictamente lineal, modela relaciones curvas usando potencias de las variables independientes."], key="text_d5")

        with ui.card(key="card_d3"):
            ui.element("span", children=["⚡ Ventajas Principales"], className="text-gray-700 font-semibold text-lg", key="label_d3")
            ui.element("p", children=["Simplicidad: Fácil de implementar, entrenar e interpretar. Los coeficientes muestran claramente el impacto de cada variable."], key="text_d6")
            ui.element("p", children=["Eficiencia: Requiere pocos recursos computacionales y es muy rápida en entrenamiento y predicción, ideal para grandes volúmenes de datos."], key="text_d7")
            ui.element("p", children=["Interpretabilidad: Los resultados son fácilmente comprensibles. La pendiente indica la magnitud y dirección de la relación."], key="text_d8")
            ui.element("p", children=["Base sólida: Sirve como punto de partida para modelos más complejos y es ampliamente utilizada en estadística y ciencia."], key="text_d9")

        with ui.card(key="card_d4"):
            ui.element("span", children=["🔍 Supuestos del Modelo"], className="text-gray-700 font-semibold text-lg", key="label_d4")
            ui.element("p", children=["Linealidad: Existe una relación lineal entre las variables independientes y la dependiente."], key="text_d10")
            ui.element("p", children=["Independencia: Las observaciones son independientes entre sí, sin correlaciones ocultas."], key="text_d11")
            ui.element("p", children=["Homoscedasticidad: La varianza de los errores es constante en todos los niveles de las variables independientes."], key="text_d12")
            ui.element("p", children=["Normalidad: Los residuos siguen una distribución normal, especialmente importante para inferencia estadística."], key="text_d13")
            ui.element("p", children=["No multicolinealidad: En regresión múltiple, las variables independientes no deben estar altamente correlacionadas entre sí."], key="text_d14")

        with ui.card(key="card_d5"):
            ui.element("span", children=["📊 Método de Mínimos Cuadrados"], className="text-gray-700 font-semibold text-lg", key="label_d5")
            ui.element("p", children=["El método más común para ajustar la línea de regresión es minimizar la suma de los cuadrados de las diferencias entre los valores observados y predichos (residuos)."], key="text_d15")
            ui.element("p", children=["Este enfoque garantiza que la línea resultante sea la que mejor se ajusta a los datos en el sentido de menor error cuadrático medio."], key="text_d16")

        with ui.card(key="card_d6"):
            ui.element("span", children=["🎲 Métricas de Evaluación"], className="text-gray-700 font-semibold text-lg", key="label_d6")
            ui.element("p", children=["R² (Coeficiente de Determinación): Indica qué porcentaje de la variabilidad de los datos es explicada por el modelo. Valores cercanos a 1 son mejores."], key="text_d17")
            ui.element("p", children=["MSE (Error Cuadrático Medio): Promedio de los cuadrados de los errores. Valores más bajos indican mejor ajuste."], key="text_d18")
            ui.element("p", children=["RMSE (Raíz del Error Cuadrático Medio): Raíz cuadrada del MSE, en las mismas unidades que la variable objetivo."], key="text_d19")
            ui.element("p", children=["MAE (Error Absoluto Medio): Promedio de los valores absolutos de los errores, menos sensible a valores atípicos que MSE."], key="text_d20")

        with ui.card(key="card_d7"):
            ui.element("span", children=["💡 Cuándo Usar Regresión Lineal"], className="text-gray-700 font-semibold text-lg", key="label_d7")
            ui.element("p", children=["Variable objetivo continua y numérica, relación aproximadamente lineal entre variables, necesidad de interpretabilidad clara, cuando los supuestos del modelo se cumplen razonablemente, como baseline antes de probar modelos más complejos, cuando se requiere rapidez en entrenamiento y predicción."], key="text_d21")
    
    elif tabs == "Formulas":
        with ui.card(key="card_f1"):
            ui.element("span", children=["📐 Regresión Lineal Simple"], className="text-gray-700 font-semibold text-lg", key="label_f1")
            st.latex(r"y = \beta_0 + \beta_1 x")
            if st.button("📋 Copiar fórmula", key="copy_f1"):
                st.code(r"y = \beta_0 + \beta_1 x", language="latex")
            ui.element("p", children=["Variables:"], className="font-semibold mt-2", key="var_title_f1")
            ui.element("p", children=["y: Variable dependiente o variable objetivo que queremos predecir."], key="text_f1a")
            ui.element("p", children=["x: Variable independiente o predictora."], key="text_f1b")
            ui.element("p", children=["β₀: Intercepto o término constante, valor de y cuando x = 0."], key="text_f1c")
            ui.element("p", children=["β₁: Pendiente o coeficiente, indica cuánto cambia y por cada unidad de cambio en x."], key="text_f1d")
            ui.element("p", children=["Propósito: Modelo básico que describe una relación lineal entre dos variables mediante una línea recta."], className="text-gray-600 italic mt-2", key="text_f1e")

        with ui.card(key="card_f2"):
            ui.element("span", children=["🎯 Regresión Lineal Múltiple"], className="text-gray-700 font-semibold text-lg", key="label_f2")
            st.latex(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n")
            if st.button("📋 Copiar fórmula", key="copy_f2"):
                st.code(r"y = \beta_0 + \beta_1 x_1 + \beta_2 x_2 + ... + \beta_n x_n", language="latex")
            ui.element("p", children=["Variables:"], className="font-semibold mt-2", key="var_title_f2")
            ui.element("p", children=["y: Variable dependiente que se predice."], key="text_f2a")
            ui.element("p", children=["x₁, x₂, ..., xₙ: Variables independientes o características de entrada."], key="text_f2b")
            ui.element("p", children=["β₀: Intercepto del modelo."], key="text_f2c")
            ui.element("p", children=["β₁, β₂, ..., βₙ: Coeficientes que indican la contribución de cada variable independiente."], key="text_f2d")
            ui.element("p", children=["n: Número de variables independientes."], key="text_f2e")
            ui.element("p", children=["Propósito: Extiende la regresión simple a múltiples variables predictoras para modelar relaciones más complejas."], className="text-gray-600 italic mt-2", key="text_f2f")

        with ui.card(key="card_f3"):
            ui.element("span", children=["📊 Función de Coste (MSE)"], className="text-gray-700 font-semibold text-lg", key="label_f3")
            st.latex(r"MSE = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2")
            if st.button("📋 Copiar fórmula", key="copy_f3"):
                st.code(r"MSE = \frac{1}{m}\sum_{i=1}^{m}(y_i - \hat{y}_i)^2", language="latex")
            ui.element("p", children=["Variables:"], className="font-semibold mt-2", key="var_title_f3")
            ui.element("p", children=["MSE: Error Cuadrático Medio (Mean Squared Error)."], key="text_f3a")
            ui.element("p", children=["m: Número total de observaciones en el conjunto de datos."], key="text_f3b")
            ui.element("p", children=["yᵢ: Valor real de la variable dependiente para la observación i."], key="text_f3c")
            ui.element("p", children=["ŷᵢ: Valor predicho por el modelo para la observación i."], key="text_f3d")
            ui.element("p", children=["Propósito: Mide el error promedio del modelo elevando al cuadrado las diferencias entre predicciones y valores reales. Valores más bajos indican mejor ajuste."], className="text-gray-600 italic mt-2", key="text_f3e")

        with ui.card(key="card_f4"):
            ui.element("span", children=["🔄 Coeficiente de Pendiente"], className="text-gray-700 font-semibold text-lg", key="label_f4")
            st.latex(r"\beta_1 = \frac{\sum_{i=1}^{m}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{m}(x_i - \bar{x})^2}")
            if st.button("📋 Copiar fórmula", key="copy_f4"):
                st.code(r"\beta_1 = \frac{\sum_{i=1}^{m}(x_i - \bar{x})(y_i - \bar{y})}{\sum_{i=1}^{m}(x_i - \bar{x})^2}", language="latex")
            ui.element("p", children=["Variables:"], className="font-semibold mt-2", key="var_title_f4")
            ui.element("p", children=["β₁: Coeficiente de la pendiente en regresión simple."], key="text_f4a")
            ui.element("p", children=["xᵢ, yᵢ: Valores individuales de las variables x e y."], key="text_f4b")
            ui.element("p", children=["x̄: Media de los valores de x."], key="text_f4c")
            ui.element("p", children=["ȳ: Media de los valores de y."], key="text_f4d")
            ui.element("p", children=["m: Número de observaciones."], key="text_f4e")
            ui.element("p", children=["Propósito: Calcula la pendiente óptima que minimiza el error cuadrático mediante el método de mínimos cuadrados ordinarios."], className="text-gray-600 italic mt-2", key="text_f4f")

        with ui.card(key="card_f5"):
            ui.element("span", children=["🎲 Coeficiente de Intercepto"], className="text-gray-700 font-semibold text-lg", key="label_f5")
            st.latex(r"\beta_0 = \bar{y} - \beta_1\bar{x}")
            if st.button("📋 Copiar fórmula", key="copy_f5"):
                st.code(r"\beta_0 = \bar{y} - \beta_1\bar{x}", language="latex")
            ui.element("p", children=["Variables:"], className="font-semibold mt-2", key="var_title_f5")
            ui.element("p", children=["β₀: Intercepto del modelo (valor de y cuando x = 0)."], key="text_f5a")
            ui.element("p", children=["ȳ: Media de los valores de y."], key="text_f5b")
            ui.element("p", children=["β₁: Pendiente calculada previamente."], key="text_f5c")
            ui.element("p", children=["x̄: Media de los valores de x."], key="text_f5d")
            ui.element("p", children=["Propósito: Calcula el intercepto asegurando que la línea de regresión pase por el punto medio (x̄, ȳ) de los datos."], className="text-gray-600 italic mt-2", key="text_f5e")

        with ui.card(key="card_f6"):
            ui.element("span", children=["✅ Coeficiente de Determinación (R²)"], className="text-gray-700 font-semibold text-lg", key="label_f6")
            st.latex(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{m}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{m}(y_i - \bar{y})^2}")
            if st.button("📋 Copiar fórmula", key="copy_f6"):
                st.code(r"R^2 = 1 - \frac{SS_{res}}{SS_{tot}} = 1 - \frac{\sum_{i=1}^{m}(y_i - \hat{y}_i)^2}{\sum_{i=1}^{m}(y_i - \bar{y})^2}", language="latex")
            ui.element("p", children=["Variables:"], className="font-semibold mt-2", key="var_title_f6")
            ui.element("p", children=["R²: Coeficiente de determinación, varía entre 0 y 1."], key="text_f6a")
            ui.element("p", children=["SSres: Suma de cuadrados de los residuos (error del modelo)."], key="text_f6b")
            ui.element("p", children=["SStot: Suma total de cuadrados (variabilidad total de los datos)."], key="text_f6c")
            ui.element("p", children=["yᵢ: Valor real."], key="text_f6d")
            ui.element("p", children=["ŷᵢ: Valor predicho."], key="text_f6e")
            ui.element("p", children=["ȳ: Media de y."], key="text_f6f")
            ui.element("p", children=["Propósito: Indica qué proporción de la variabilidad de la variable dependiente es explicada por el modelo. Valores cercanos a 1 indican mejor ajuste."], className="text-gray-600 italic mt-2", key="text_f6g")











            
            
            
            
            
            
            
            
            
            

elif st.session_state.opcion == "Ejemplos_Regresion_Lineal":
    st.title("📈 Predicción Continua con Regresión Lineal - Fórmula 1")
    st.markdown("### Ejemplo práctico: Predecir el total de vueltas de una carrera")

    with st.spinner("Cargando datos de la API..."):
        try:
            df_raw = Getdata.get_dataset()
            st.success("✅ Datos cargados exitosamente")

            with ui.card(key="lr_card_step1"):
                st.markdown("#### 📊 Paso 1: Exploración del Dataset")
                st.dataframe(df_raw.head(10), use_container_width=True)
                st.markdown(f"**Total de registros:** {len(df_raw)}")
                st.markdown(f"**Columnas:** {', '.join(df_raw.columns.tolist())}")
                st.code("""
df = Getdata.get_dataset()
df.head()
                """)

            with ui.card(key="lr_card_step2"):
                st.markdown("#### 🧹 Paso 2: Limpieza de Datos")
                df_clean = Getdata.cleanData(df_raw)
                df_clean = df_clean[df_clean['laps_total'].notna()]
                st.dataframe(df_clean.head(10), use_container_width=True)
                st.code("""
df = Getdata.cleanData(df)
df = df[df['laps_total'].notna()]
                """)

            with ui.card(key="lr_card_step3"):
                st.markdown("#### 🎯 Paso 3: Variable Objetivo")
                st.markdown("La variable objetivo será **laps_total**")
                st.metric("Promedio de Vueltas", f"{df_clean['laps_total'].mean():.2f}")
                st.metric("Máximo de Vueltas", df_clean['laps_total'].max())
                st.code("""
y = df['laps_total']
                """)

            with ui.card(key="lr_card_step4"):
                st.markdown("#### 🔧 Paso 4: Ingeniería de Características")

                le_circuit = LabelEncoder()
                le_country = LabelEncoder()
                le_status = LabelEncoder()

                df_model = df_clean.copy()
                df_model['circuit_encoded'] = le_circuit.fit_transform(df_model['circuit'])
                df_model['country_encoded'] = le_country.fit_transform(df_model['country'])
                df_model['status_encoded'] = le_status.fit_transform(df_model['status'])
                df_model['month'] = pd.to_datetime(df_model['date']).dt.month

                features = ['circuit_encoded', 'country_encoded', 'status_encoded', 'month']
                target = 'laps_total'

                st.dataframe(df_model[features + [target]].head(10), use_container_width=True)
                st.code("""
df['circuit_encoded'] = LabelEncoder().fit_transform(df['circuit'])
df['country_encoded'] = LabelEncoder().fit_transform(df['country'])
df['status_encoded'] = LabelEncoder().fit_transform(df['status'])
df['month'] = pd.to_datetime(df['date']).dt.month
X = df[['circuit_encoded','country_encoded','status_encoded','month']]
                """)

            with ui.card(key="lr_card_step5"):
                st.markdown("#### 📐 Paso 5: División de Datos")

                X = df_model[features]
                y = df_model[target]

                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Entrenamiento", len(X_train))
                with col2:
                    st.metric("Prueba", len(X_test))
                st.code("""
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
                """)

            with ui.card(key="lr_card_step6"):
                st.markdown("#### 🤖 Paso 6: Entrenamiento del Modelo")

                model = LinearRegression()
                model.fit(X_train, y_train)

                st.success("✅ Modelo entrenado exitosamente")

                coef_df = pd.DataFrame({
                    "Característica": features,
                    "Coeficiente": model.coef_
                }).sort_values("Coeficiente", key=abs, ascending=False)

                st.dataframe(coef_df, use_container_width=True)
                st.markdown(f"**Intercepto:** {model.intercept_:.4f}")
                st.code("""
model = LinearRegression()
model.fit(X_train, y_train)
                """)

            with ui.card(key="lr_card_step7"):
                st.markdown("#### 📊 Paso 7: Evaluación del Modelo")

                y_pred = model.predict(X_test)

                mae = mean_absolute_error(y_test, y_pred)
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                r2 = r2_score(y_test, y_pred)

                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("MAE", f"{mae:.2f}")
                with col2:
                    st.metric("MSE", f"{mse:.2f}")
                with col3:
                    st.metric("RMSE", f"{rmse:.2f}")
                with col4:
                    st.metric("R²", f"{r2:.3f}")

                df_compare = pd.DataFrame({
                    "Real": y_test.values[:10],
                    "Predicho": y_pred[:10]
                })
                st.dataframe(df_compare, use_container_width=True)
                st.code("""
y_pred = model.predict(X_test)
mean_absolute_error(y_test, y_pred)
mean_squared_error(y_test, y_pred)
r2_score(y_test, y_pred)
                """)

            with ui.card(key="lr_card_step8"):
                st.markdown("#### 🔮 Paso 8: Predicción Directa")

                ejemplo_idx = X_test.index[0]
                ejemplo = X_test.loc[[ejemplo_idx]]
                ejemplo_real = y_test.loc[ejemplo_idx]

                prediccion = model.predict(ejemplo)[0]

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Vueltas Reales", f"{ejemplo_real:.0f}")
                with col2:
                    st.metric("Vueltas Predichas", f"{prediccion:.0f}")
                with col3:
                    st.metric("Error Absoluto", f"{abs(ejemplo_real - prediccion):.0f}")

                y_calc = model.intercept_
                partes = [f"{model.intercept_:.4f}"]
                for i, feat in enumerate(features):
                    valor = ejemplo[feat].values[0]
                    coef = model.coef_[i]
                    y_calc += coef * valor
                    partes.append(f"{coef:.4f} \\times {valor:.2f}")

                st.latex("y = " + " + ".join(partes))
                st.latex(f"y = {y_calc:.4f}")

                st.code("""
ejemplo = X_test.iloc[[0]]
model.predict(ejemplo)
                """)

        except Exception as e:
            st.error(f"Error al cargar los datos: {str(e)}")
            st.info("Verifica la API y la conexión a internet")































































elif st.session_state.opcion == "Calculadora Regresion Linear":
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

        div[data-testid="stFileUploader"]{
            background:var(--card);
            border-radius:12px;
            padding:10px;
        }

        div[data-testid="stDataFrame"]{
            background:var(--card);
            border-radius:12px;
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

        .explicacion-box{
            background:var(--card);
            border-left:4px solid var(--primary);
            padding:20px;
            border-radius:12px;
            margin:20px 0;
        }

        .metric-destacado{
            background:linear-gradient(135deg, var(--primary), var(--accent));
            padding:15px;
            border-radius:10px;
            text-align:center;
            font-size:1.2em;
            font-weight:bold;
        }
        </style>
    """, unsafe_allow_html=True)

    st.title("📊 Calculadora de Regresión Lineal")
    st.markdown("### Carga tu dataset y entrena tu modelo predictivo")

    archivo_cargado = st.file_uploader("📁 Sube tu archivo CSV o Excel", type=['csv', 'xlsx', 'xls'], key="uploader_lineal")

    if archivo_cargado is not None:
        try:
            if archivo_cargado.name.endswith('.csv'):
                datos = pd.read_csv(archivo_cargado)
            else:
                datos = pd.read_excel(archivo_cargado)

            with ui.card(key="linear_vista_previa"):
                st.markdown("#### 📊 Vista previa del Dataset")
                st.dataframe(datos.head(10), use_container_width=True)

                c1, c2, c3 = st.columns(3)
                with c1:
                    st.metric("Filas", datos.shape[0])
                with c2:
                    st.metric("Columnas", datos.shape[1])
                with c3:
                    st.metric("Valores Nulos", datos.isnull().sum().sum())

            with ui.card(key="linear_limpieza_datos"):
                st.markdown("#### 🧹 Limpieza de Datos")

                c1, c2, c3 = st.columns(3)
                with c1:
                    eliminar_duplicados = st.checkbox("Eliminar filas duplicadas", True)
                with c2:
                    manejar_nulos = st.selectbox("Manejar valores nulos", ["Eliminar filas con nulos", "Rellenar con media/moda", "No hacer nada"])
                with c3:
                    eliminar_outliers = st.checkbox("Eliminar outliers", False)

                if st.button("Aplicar Limpieza", use_container_width=True):
                    original = datos.shape[0]

                    if eliminar_duplicados:
                        datos = datos.drop_duplicates()

                    if manejar_nulos == "Eliminar filas con nulos":
                        datos = datos.dropna()
                    elif manejar_nulos == "Rellenar con media/moda":
                        for c in datos.columns:
                            if datos[c].dtype in ["int64","float64"]:
                                datos[c] = datos[c].fillna(datos[c].mean())
                            else:
                                datos[c] = datos[c].fillna(datos[c].mode()[0])

                    if eliminar_outliers:
                        for c in datos.select_dtypes(include=["int64","float64"]).columns:
                            q1 = datos[c].quantile(0.25)
                            q3 = datos[c].quantile(0.75)
                            iqr = q3 - q1
                            datos = datos[(datos[c] >= q1 - 1.5*iqr) & (datos[c] <= q3 + 1.5*iqr)]

                    st.success(f"Dataset limpio: {original} → {datos.shape[0]} filas")
                    st.dataframe(datos.head(), use_container_width=True)

            with ui.card(key="linear_configuracion"):
                st.markdown("#### ⚙️ Configuración del Modelo")

                columnas_numericas = datos.select_dtypes(include=["int64","float64"]).columns.tolist()
                variable_objetivo = st.selectbox("Variable objetivo", columnas_numericas)
                columnas_caracteristicas = st.multiselect("Características", [c for c in datos.columns if c != variable_objetivo])
                tamano_prueba = st.slider("Tamaño prueba (%)", 10, 50, 20)
                normalizar = st.checkbox("Normalizar datos", True)

            if variable_objetivo and columnas_caracteristicas:
                if st.button("Entrenar Modelo", use_container_width=True):
                    datos_modelo = datos.dropna(subset=[variable_objetivo] + columnas_caracteristicas)
                    X = datos_modelo[columnas_caracteristicas]
                    y = datos_modelo[variable_objetivo]

                    encoders = {}
                    for c in X.columns:
                        if X[c].dtype == "object":
                            le = LabelEncoder()
                            X[c] = le.fit_transform(X[c].astype(str))
                            encoders[c] = le

                    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=tamano_prueba/100, random_state=42)

                    scaler = None
                    if normalizar:
                        scaler = StandardScaler()
                        X_train = scaler.fit_transform(X_train)
                        X_test = scaler.transform(X_test)

                    modelo = LinearRegression()
                    modelo.fit(X_train, y_train)

                    st.success("✅ Modelo entrenado correctamente")

                    # Predicciones
                    y_pred_train = modelo.predict(X_train)
                    y_pred_test = modelo.predict(X_test)

                    # Métricas
                    r2_train = r2_score(y_train, y_pred_train)
                    r2_test = r2_score(y_test, y_pred_test)
                    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
                    rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))
                    mae_test = np.mean(np.abs(y_test - y_pred_test))

                    # Mostrar métricas principales
                    with ui.card(key="metricas_principales"):
                        st.markdown("#### 📈 Métricas del Modelo")
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("R² (Test)", f"{r2_test:.4f}")
                        with col2:
                            st.metric("R² (Train)", f"{r2_train:.4f}")
                        with col3:
                            st.metric("RMSE (Test)", f"{rmse_test:.4f}")
                        with col4:
                            st.metric("MAE (Test)", f"{mae_test:.4f}")

                    # Gráfico de coeficientes
                    with ui.card(key="coeficientes"):
                        st.markdown("#### 🎯 Importancia de las Características")
                        coef_df = pd.DataFrame({
                            "Característica": columnas_caracteristicas,
                            "Coeficiente": modelo.coef_
                        }).sort_values("Coeficiente", key=np.abs, ascending=False)

                        fig_coef = px.bar(
                            coef_df,
                            x="Coeficiente",
                            y="Característica",
                            orientation="h",
                            color="Coeficiente",
                            color_continuous_scale=["#38bdf8","#a78bfa","#f43f5e"],
                            template="plotly_dark",
                            title="Coeficientes de Regresión"
                        )
                        fig_coef.update_layout(height=400)
                        st.plotly_chart(fig_coef, use_container_width=True)

                    # GRÁFICO DE LA RECTA DE REGRESIÓN
                    with ui.card(key="recta_regresion"):
                        st.markdown("#### 📉 Recta de Regresión - Valores Reales vs Predichos")
                        
                        # Crear DataFrame para el gráfico
                        df_plot = pd.DataFrame({
                            'Real': y_test,
                            'Predicho': y_pred_test
                        })
                        
                        # Crear figura con scatter plot
                        fig_regression = go.Figure()
                        
                        # Puntos reales vs predichos
                        fig_regression.add_trace(go.Scatter(
                            x=y_test,
                            y=y_pred_test,
                            mode='markers',
                            name='Predicciones',
                            marker=dict(
                                size=8,
                                color=y_pred_test,
                                colorscale='Viridis',
                                showscale=True,
                                colorbar=dict(title="Valor Predicho"),
                                line=dict(width=1, color='white')
                            ),
                            text=[f'Real: {r:.2f}<br>Pred: {p:.2f}<br>Error: {abs(r-p):.2f}' 
                                  for r, p in zip(y_test, y_pred_test)],
                            hovertemplate='<b>%{text}</b><extra></extra>'
                        ))
                        
                        # Línea de regresión perfecta (diagonal)
                        min_val = min(y_test.min(), y_pred_test.min())
                        max_val = max(y_test.max(), y_pred_test.max())
                        fig_regression.add_trace(go.Scatter(
                            x=[min_val, max_val],
                            y=[min_val, max_val],
                            mode='lines',
                            name='Predicción Perfecta',
                            line=dict(color='#f43f5e', width=3, dash='dash')
                        ))
                        
                        # Línea de tendencia (regresión de las predicciones)
                        z = np.polyfit(y_test, y_pred_test, 1)
                        p = np.poly1d(z)
                        fig_regression.add_trace(go.Scatter(
                            x=sorted(y_test),
                            y=p(sorted(y_test)),
                            mode='lines',
                            name=f'Tendencia (y={z[0]:.3f}x+{z[1]:.3f})',
                            line=dict(color='#38bdf8', width=2)
                        ))
                        
                        fig_regression.update_layout(
                            template='plotly_dark',
                            height=500,
                            xaxis_title='Valores Reales',
                            yaxis_title='Valores Predichos',
                            hovermode='closest',
                            showlegend=True,
                            legend=dict(
                                yanchor="top",
                                y=0.99,
                                xanchor="left",
                                x=0.01,
                                bgcolor="rgba(0,0,0,0.5)"
                            )
                        )
                        
                        st.plotly_chart(fig_regression, use_container_width=True)

                    # EXPLICACIÓN DETALLADA DEL GRÁFICO
                    with ui.card(key="explicacion_grafico"):
                        st.markdown("#### 📚 Interpretación del Gráfico y Métricas")
                        
                        st.markdown(f"""
                        <div class="explicacion-box">
                        <h4>🎯 ¿Qué muestra este gráfico?</h4>
                        <p>Este gráfico compara los <b>valores reales</b> (eje X) con los <b>valores predichos por el modelo</b> (eje Y). 
                        Cada punto representa una observación del conjunto de prueba.</p>
                        
                        <h4>📏 Elementos del Gráfico:</h4>
                        <ul>
                            <li><b>Puntos de colores:</b> Cada punto es una predicción. Mientras más cerca esté de la línea roja discontinua, mejor es la predicción.</li>
                            <li><b>Línea roja discontinua (y=x):</b> Representa la predicción perfecta. Si todos los puntos estuvieran en esta línea, el modelo sería perfecto.</li>
                            <li><b>Línea azul continua:</b> Es la línea de tendencia de tus predicciones. Muestra la relación general entre valores reales y predichos.</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Análisis de las métricas
                        st.markdown(f"""
                        <div class="explicacion-box">
                        <h4>📊 Análisis de tus Métricas:</h4>
                        
                        <p><b>🎯 R² (Coeficiente de Determinación) = {r2_test:.4f}</b></p>
                        <ul>
                            <li>Indica qué porcentaje de la variabilidad de los datos es explicado por el modelo.</li>
                            <li><b>Tu modelo explica el {r2_test*100:.2f}% de la variación</b> en {variable_objetivo}.</li>
                            <li>Interpretación: {'🟢 Excelente' if r2_test > 0.9 else '🟡 Bueno' if r2_test > 0.7 else '🟠 Aceptable' if r2_test > 0.5 else '🔴 Necesita mejorar'}</li>
                        </ul>
                        
                        <p><b>📏 RMSE (Error Cuadrático Medio) = {rmse_test:.4f}</b></p>
                        <ul>
                            <li>Es el promedio de las diferencias entre valores reales y predichos.</li>
                            <li><b>En promedio, tus predicciones se desvían ±{rmse_test:.2f} unidades</b> del valor real.</li>
                            <li>Mientras más bajo, mejor es el modelo.</li>
                        </ul>
                        
                        <p><b>📐 MAE (Error Absoluto Medio) = {mae_test:.4f}</b></p>
                        <ul>
                            <li>El error promedio absoluto de tus predicciones.</li>
                            <li><b>Tus predicciones tienen un error promedio de {mae_test:.2f} unidades.</b></li>
                        </ul>
                        
                        <p><b>🔍 Diferencia Train vs Test:</b></p>
                        <ul>
                            <li>R² Train: {r2_train:.4f} | R² Test: {r2_test:.4f}</li>
                            <li>Diferencia: {abs(r2_train - r2_test):.4f}</li>
                            <li>{'🟢 Modelo bien generalizado' if abs(r2_train - r2_test) < 0.1 else '🟡 Posible sobreajuste' if r2_train > r2_test + 0.1 else '🟠 Revisar modelo'}</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        # Ecuación del modelo
                        st.markdown("#### 🧮 Ecuación del Modelo")
                        ecuacion = f"{variable_objetivo} = {modelo.intercept_:.4f}"
                        for i, feat in enumerate(columnas_caracteristicas):
                            coef = modelo.coef_[i]
                            ecuacion += f" + ({coef:.4f} × {feat})"
                        
                        st.code(ecuacion, language="text")
                        
                        st.markdown(f"""
                        <div class="explicacion-box">
                        <h4>💡 Recomendaciones:</h4>
                        <ul>
                            <li>{'✅ Tu modelo tiene buen rendimiento.' if r2_test > 0.7 else '⚠️ Considera agregar más características o usar otro algoritmo.'}</li>
                            <li>{'✅ No hay sobreajuste significativo.' if abs(r2_train - r2_test) < 0.1 else '⚠️ Hay sobreajuste, considera regularización o más datos.'}</li>
                            <li>Los coeficientes más altos (en valor absoluto) son las características más influyentes.</li>
                        </ul>
                        </div>
                        """, unsafe_allow_html=True)

                    # Gráfico de residuos
                    with ui.card(key="residuos"):
                        st.markdown("#### 📊 Análisis de Residuos")
                        residuos = y_test - y_pred_test
                        
                        fig_residuos = go.Figure()
                        fig_residuos.add_trace(go.Scatter(
                            x=y_pred_test,
                            y=residuos,
                            mode='markers',
                            marker=dict(
                                size=8,
                                color=residuos,
                                colorscale='RdYlGn_r',
                                showscale=True,
                                colorbar=dict(title="Residuo")
                            ),
                            name='Residuos'
                        ))
                        
                        fig_residuos.add_hline(y=0, line_dash="dash", line_color="red")
                        fig_residuos.update_layout(
                            template='plotly_dark',
                            height=400,
                            xaxis_title='Valores Predichos',
                            yaxis_title='Residuos',
                            title='Residuos vs Valores Predichos'
                        )
                        
                        st.plotly_chart(fig_residuos, use_container_width=True)
                        
                        st.info("ℹ️ Los residuos deberían distribuirse aleatoriamente alrededor de cero. Un patrón sistemático indica problemas en el modelo.")

        except Exception as e:
            st.error(f"❌ Error: {str(e)}")
    else:
        st.info("📁 Carga un archivo CSV o Excel para comenzar el análisis")