# ============================================================
# 05_streamlit_dashboard.py
# Herdez Smart Supply - Guía Interactiva de Supply Chain
# ============================================================
# Objetivo:
# Convertir los datos de inventario y demanda en una experiencia visual,
# clara y entendible para usuarios que NO conocen Supply Chain.
#
# Este archivo está diseñado para:
# - Cargar datos automáticamente desde rutas comunes.
# - Detectar columnas aunque tengan nombres distintos.
# - Crear métricas simples de inventario.
# - Mostrar filtros amigables.
# - Explicar cada gráfico con lenguaje sencillo.
# - Mostrar alertas visuales de desabasto, riesgo y exceso.
# ============================================================


import streamlit as st
import pandas as pd
import plotly.express as px
from pathlib import Path
from datetime import datetime


# ============================================================
# CONFIGURACIÓN GENERAL DE LA APP
# ============================================================

st.set_page_config(
    page_title="Herdez Smart Supply | Guía Interactiva",
    page_icon="📦",
    layout="wide"
)


# ============================================================
# ESTILOS VISUALES SIMPLES
# ============================================================

st.markdown(
    """
    <style>
        .main-title {
            font-size: 2.3rem;
            font-weight: 800;
            margin-bottom: 0.3rem;
        }

        .subtitle {
            font-size: 1.1rem;
            color: #555;
            margin-bottom: 1.5rem;
        }

        .section-card {
            background-color: #f7f9fb;
            padding: 1.2rem;
            border-radius: 14px;
            border: 1px solid #e6eaf0;
            margin-bottom: 1rem;
        }

        .simple-text {
            font-size: 1rem;
            line-height: 1.6;
        }

        .small-muted {
            color: #666;
            font-size: 0.9rem;
        }
    </style>
    """,
    unsafe_allow_html=True
)


# ============================================================
# FUNCIONES AUXILIARES
# ============================================================

def normalize_text(value):
    """
    Normaliza texto para comparar columnas sin depender de mayúsculas,
    espacios o guiones.
    """
    return str(value).lower().strip().replace(" ", "_").replace("-", "_")


def find_column(df, possible_names):
    """
    Busca una columna en el DataFrame usando una lista de posibles nombres.

    Esto evita que la app se rompa si el dataset usa nombres como:
    - product
    - producto
    - product_name
    - sku
    """

    normalized_columns = {
        normalize_text(col): col
        for col in df.columns
    }

    for name in possible_names:
        normalized_name = normalize_text(name)

        if normalized_name in normalized_columns:
            return normalized_columns[normalized_name]

    return None


@st.cache_data
def load_data():
    """
    Carga los datos desde rutas comunes del proyecto.

    Puedes ajustar esta lista si tu archivo se llama diferente.
    """

    possible_paths = [
        "data/supply_chain_data.csv",
        "data/final_dataset.csv",
        "data/inventory_data.csv",
        "data/herdez_supply_chain.csv",
        "outputs/final_dataset.csv",
        "outputs/supply_chain_data.csv",
        "outputs/inventory_data.csv",
        "final_dataset.csv",
        "supply_chain_data.csv",
        "inventory_data.csv"
    ]

    for path in possible_paths:
        file_path = Path(path)

        if file_path.exists():
            if file_path.suffix.lower() == ".csv":
                return pd.read_csv(file_path), str(file_path)

            if file_path.suffix.lower() in [".xlsx", ".xls"]:
                return pd.read_excel(file_path), str(file_path)

            if file_path.suffix.lower() == ".parquet":
                return pd.read_parquet(file_path), str(file_path)

    return None, None


def create_demo_data():
    """
    Crea datos de ejemplo si no se encuentra ningún archivo.

    Esto sirve para que el dashboard pueda abrirse y demostrarse,
    aunque todavía no exista un dataset cargado.
    """

    data = {
        "producto": [
            "Salsa Casera", "Salsa Verde", "Champiñones", "Atún", "Mermelada",
            "Salsa Casera", "Salsa Verde", "Champiñones", "Atún", "Mermelada",
            "Salsa Casera", "Salsa Verde", "Champiñones", "Atún", "Mermelada"
        ],
        "region": [
            "Centro", "Centro", "Centro", "Centro", "Centro",
            "Norte", "Norte", "Norte", "Norte", "Norte",
            "Sur", "Sur", "Sur", "Sur", "Sur"
        ],
        "stock_actual": [
            120, 45, 300, 80, 600,
            70, 160, 40, 500, 90,
            20, 250, 180, 60, 700
        ],
        "demanda_pronosticada": [
            100, 90, 120, 100, 150,
            120, 110, 80, 140, 130,
            110, 100, 160, 100, 180
        ],
        "ventas": [
            95, 85, 110, 98, 145,
            115, 108, 75, 135, 125,
            105, 95, 150, 95, 175
        ],
        "fecha": pd.date_range(start="2026-01-01", periods=15, freq="W")
    }

    return pd.DataFrame(data)


def safe_numeric(series):
    """
    Convierte una columna a número sin romper la app.
    Los valores inválidos se convierten en 0.
    """
    return pd.to_numeric(series, errors="coerce").fillna(0)


def classify_inventory(stock, demand):
    """
    Clasifica el estado del inventario de forma sencilla.

    La lógica es:
    - Desabasto: no alcanza para cubrir la demanda.
    - Riesgo: alcanza, pero con muy poco colchón.
    - Saludable: parece suficiente.
    - Exceso: hay demasiado inventario comparado con la demanda.
    """

    if demand <= 0:
        if stock > 0:
            return "Exceso"
        return "Sin datos"

    ratio = stock / demand

    if ratio < 1:
        return "Desabasto"
    elif ratio < 1.3:
        return "Riesgo"
    elif ratio <= 2.5:
        return "Saludable"
    else:
        return "Exceso"


def status_priority(status):
    """
    Ordena los estados para que los más críticos aparezcan primero.
    """
    order = {
        "Desabasto": 1,
        "Riesgo": 2,
        "Exceso": 3,
        "Saludable": 4,
        "Sin datos": 5
    }

    return order.get(status, 99)


def format_number(value):
    """
    Formato simple para números grandes.
    """
    try:
        return f"{value:,.0f}"
    except Exception:
        return value


# ============================================================
# ENCABEZADO
# ============================================================

st.markdown('<div class="main-title">📦 Herdez Smart Supply</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="subtitle">Una guía visual para entender inventario, demanda y riesgos de abastecimiento.</div>',
    unsafe_allow_html=True
)


# ============================================================
# CARGA DE DATOS
# ============================================================

df, source_path = load_data()

if df is None:
    st.warning(
        "No encontré automáticamente un archivo de datos. "
        "Puedes cargar uno manualmente o usar datos de demostración."
    )

    uploaded_file = st.file_uploader(
        "Carga tu archivo CSV o Excel",
        type=["csv", "xlsx", "xls"]
    )

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        else:
            df = pd.read_excel(uploaded_file)

        source_path = uploaded_file.name

    else:
        st.info("Usaré datos de demostración para que puedas ver cómo funciona el dashboard.")
        df = create_demo_data()
        source_path = "Datos de demostración"


# ============================================================
# DETECCIÓN AUTOMÁTICA DE COLUMNAS
# ============================================================

product_col = find_column(
    df,
    [
        "producto", "product", "product_name", "nombre_producto",
        "sku", "item", "material", "articulo", "artículo"
    ]
)

region_col = find_column(
    df,
    [
        "region", "región", "zona", "state", "estado",
        "location", "ubicacion", "ubicación", "warehouse", "almacen", "almacén"
    ]
)

stock_col = find_column(
    df,
    [
        "stock_actual", "stock", "inventory", "inventory_level",
        "inventario", "inventario_actual", "current_stock",
        "available_stock", "existencias"
    ]
)

demand_col = find_column(
    df,
    [
        "demanda_pronosticada", "demanda", "demand",
        "predicted_demand", "forecast", "forecast_demand",
        "pronostico", "pronóstico", "demanda_estimada"
    ]
)

sales_col = find_column(
    df,
    [
        "ventas", "sales", "units_sold", "unidades_vendidas",
        "sell_out", "venta"
    ]
)

date_col = find_column(
    df,
    [
        "fecha", "date", "week", "semana", "month",
        "mes", "periodo", "period"
    ]
)

status_existing_col = find_column(
    df,
    [
        "estado_inventario", "inventory_status", "status",
        "estado", "stock_status"
    ]
)


# ============================================================
# VALIDACIÓN MÍNIMA
# ============================================================

required_missing = []

if product_col is None:
    required_missing.append("producto")

if stock_col is None:
    required_missing.append("stock o inventario")

if demand_col is None:
    required_missing.append("demanda o pronóstico")

if required_missing:
    st.error(
        "No pude identificar columnas esenciales en el archivo: "
        + ", ".join(required_missing)
    )

    st.write("Columnas disponibles en tu archivo:")
    st.write(list(df.columns))

    st.stop()


# ============================================================
# PREPARACIÓN DE DATOS
# ============================================================

df = df.copy()

df[stock_col] = safe_numeric(df[stock_col])
df[demand_col] = safe_numeric(df[demand_col])

if sales_col is not None:
    df[sales_col] = safe_numeric(df[sales_col])

if date_col is not None:
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")

# Si no existe región, se crea una columna genérica.
if region_col is None:
    df["region_detectada"] = "Todas las regiones"
    region_col = "region_detectada"

# Si no existe estado de inventario, lo calculamos.
if status_existing_col is None:
    df["estado_inventario_calculado"] = df.apply(
        lambda row: classify_inventory(row[stock_col], row[demand_col]),
        axis=1
    )
    status_col = "estado_inventario_calculado"
else:
    status_col = status_existing_col

# Métrica central: cobertura de inventario.
# Se interpreta como cuántas veces el stock cubre la demanda esperada.
df["cobertura_inventario"] = df.apply(
    lambda row: row[stock_col] / row[demand_col] if row[demand_col] > 0 else 0,
    axis=1
)

# Brecha entre stock y demanda.
# Positivo: sobra stock.
# Negativo: falta stock.
df["brecha_stock_demanda"] = df[stock_col] - df[demand_col]


# ============================================================
# INTRODUCCIÓN INTUITIVA
# ============================================================

st.markdown(
    """
    <div class="section-card">
        <h3>🧭 ¿Qué estamos viendo aquí?</h3>
        <p class="simple-text">
            Imagina que el inventario es como la despensa de una tienda.
            Si hay muy poco producto, los clientes llegan y no encuentran lo que buscan.
            Eso significa ventas perdidas.  
            Pero si hay demasiado producto, el dinero se queda atrapado en cajas,
            anaqueles o almacenes, y algunos productos pueden caducar o generar costos extra.
        </p>
        <p class="simple-text">
            Esta guía te ayuda a responder tres preguntas sencillas:
        </p>
        <ul class="simple-text">
            <li>¿Tenemos suficiente inventario para cubrir la demanda?</li>
            <li>¿Dónde existe riesgo de quedarnos sin producto?</li>
            <li>¿Dónde tenemos demasiado inventario detenido?</li>
        </ul>
    </div>
    """,
    unsafe_allow_html=True
)

st.caption(f"Fuente de datos: {source_path}")


# ============================================================
# SIDEBAR: FILTROS AMIGABLES
# ============================================================

st.sidebar.header("🔎 Explora paso a paso")

st.sidebar.markdown(
    """
    Usa estos filtros como si fueran una lupa.
    Puedes enfocarte en un producto, una región o un estado del inventario.
    """
)

product_options = ["Todos"] + sorted(df[product_col].dropna().astype(str).unique().tolist())
selected_product = st.sidebar.selectbox(
    "Producto",
    product_options
)

region_options = ["Todas"] + sorted(df[region_col].dropna().astype(str).unique().tolist())
selected_region = st.sidebar.selectbox(
    "Región / ubicación",
    region_options
)

status_options = ["Todos"] + sorted(
    df[status_col].dropna().astype(str).unique().tolist(),
    key=status_priority
)
selected_status = st.sidebar.selectbox(
    "Estado del inventario",
    status_options
)

min_coverage = float(df["cobertura_inventario"].min())
max_coverage = float(df["cobertura_inventario"].max())

coverage_range = st.sidebar.slider(
    "Rango de cobertura de inventario",
    min_value=round(min_coverage, 2),
    max_value=round(max_coverage, 2) if max_coverage > min_coverage else round(min_coverage + 1, 2),
    value=(
        round(min_coverage, 2),
        round(max_coverage, 2) if max_coverage > min_coverage else round(min_coverage + 1, 2)
    )
)

st.sidebar.info(
    "La cobertura indica cuántas veces el inventario alcanza para cubrir la demanda. "
    "Ejemplo: 1.0 significa que el stock apenas cubre la demanda esperada."
)


# ============================================================
# APLICACIÓN DE FILTROS
# ============================================================

filtered_df = df.copy()

if selected_product != "Todos":
    filtered_df = filtered_df[
        filtered_df[product_col].astype(str) == selected_product
    ]

if selected_region != "Todas":
    filtered_df = filtered_df[
        filtered_df[region_col].astype(str) == selected_region
    ]

if selected_status != "Todos":
    filtered_df = filtered_df[
        filtered_df[status_col].astype(str) == selected_status
    ]

filtered_df = filtered_df[
    (filtered_df["cobertura_inventario"] >= coverage_range[0]) &
    (filtered_df["cobertura_inventario"] <= coverage_range[1])
]


# ============================================================
# MENSAJE SI LOS FILTROS NO TRAEN DATOS
# ============================================================

if filtered_df.empty:
    st.warning(
        "No hay datos con los filtros seleccionados. "
        "Intenta ampliar el rango de cobertura o seleccionar otro producto/región."
    )
    st.stop()


# ============================================================
# RESUMEN EJECUTIVO CON ALERTAS VISUALES
# ============================================================

st.header("🚦 Resumen rápido del inventario")

total_stock = filtered_df[stock_col].sum()
total_demand = filtered_df[demand_col].sum()
coverage_avg = filtered_df["cobertura_inventario"].mean()
stock_gap = total_stock - total_demand

desabasto_count = (filtered_df[status_col].astype(str) == "Desabasto").sum()
riesgo_count = (filtered_df[status_col].astype(str) == "Riesgo").sum()
exceso_count = (filtered_df[status_col].astype(str) == "Exceso").sum()
saludable_count = (filtered_df[status_col].astype(str) == "Saludable").sum()

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric(
        label="Inventario total",
        value=format_number(total_stock),
        help="Cantidad total de producto disponible en el filtro seleccionado."
    )

with col2:
    st.metric(
        label="Demanda esperada",
        value=format_number(total_demand),
        help="Cantidad que se espera vender o necesitar."
    )

with col3:
    st.metric(
        label="Brecha stock vs demanda",
        value=format_number(stock_gap),
        delta="Sobra stock" if stock_gap >= 0 else "Falta stock",
        help="Stock menos demanda. Si es negativo, no alcanza el inventario."
    )

with col4:
    st.metric(
        label="Cobertura promedio",
        value=f"{coverage_avg:.2f}x",
        delta="Bien" if coverage_avg >= 1.3 and coverage_avg <= 2.5 else "Revisar",
        help="Cuántas veces el inventario cubre la demanda esperada."
    )


# Alertas interpretables para usuarios no técnicos.
if desabasto_count > 0:
    st.error(
        f"🚨 Atención: hay {desabasto_count} registro(s) en desabasto. "
        "Esto significa que el inventario no alcanza para cubrir la demanda esperada."
    )
elif riesgo_count > 0:
    st.warning(
        f"⚠️ Hay {riesgo_count} registro(s) en zona de riesgo. "
        "Todavía hay inventario, pero el margen es bajo."
    )
else:
    st.success(
        "✅ No se detecta desabasto en la selección actual. "
        "El inventario parece suficiente para cubrir la demanda."
    )

if exceso_count > 0:
    st.info(
        f"📦 También hay {exceso_count} registro(s) con posible exceso de stock. "
        "Esto puede significar dinero detenido en inventario."
    )


# ============================================================
# EXPLICACIÓN DEL SEMÁFORO
# ============================================================

with st.expander("🟢🟡🔴 ¿Cómo se interpreta el semáforo de inventario?"):
    st.markdown(
        """
        - **Desabasto:** el stock es menor que la demanda esperada.  
          Es la señal más crítica porque puede provocar ventas perdidas.

        - **Riesgo:** el stock alcanza, pero con muy poco margen.  
          Si la demanda sube un poco, podríamos quedarnos sin producto.

        - **Saludable:** el inventario parece suficiente y razonable.  
          Es el punto más equilibrado.

        - **Exceso:** hay mucho más inventario del necesario.  
          No siempre es malo, pero puede significar dinero detenido, saturación de almacén o riesgo de caducidad.
        """
    )


# ============================================================
# GRÁFICO 1: COMPARACIÓN STOCK VS DEMANDA
# ============================================================

st.header("📊 1. Comparación de stock contra demanda")

grouped_product = (
    filtered_df
    .groupby(product_col, as_index=False)
    .agg({
        stock_col: "sum",
        demand_col: "sum",
        "brecha_stock_demanda": "sum",
        "cobertura_inventario": "mean"
    })
)

grouped_product = grouped_product.sort_values(
    by="brecha_stock_demanda",
    ascending=True
)

chart_data = grouped_product.melt(
    id_vars=[product_col],
    value_vars=[stock_col, demand_col],
    var_name="Métrica",
    value_name="Cantidad"
)

fig_stock_demand = px.bar(
    chart_data,
    x=product_col,
    y="Cantidad",
    color="Métrica",
    barmode="group",
    title="Stock disponible vs demanda esperada por producto",
    labels={
        product_col: "Producto",
        "Cantidad": "Cantidad",
        "Métrica": "Indicador"
    }
)

fig_stock_demand.update_layout(
    xaxis_tickangle=-35,
    legend_title_text="Indicador",
    height=480
)

st.plotly_chart(fig_stock_demand, use_container_width=True)

st.info(
    """
    **Guía de lectura:**  
    Este gráfico compara cuánto producto tienes disponible contra cuánto se espera vender o necesitar.

    - Si la barra de **stock** es más baja que la de **demanda**, puede haber desabasto.
    - Si la barra de **stock** es mucho más alta que la de **demanda**, puede haber exceso.
    - La comparación clave es visual: busca productos donde las dos barras estén muy separadas.
    """
)


# ============================================================
# GRÁFICO 2: ESTADO GENERAL DEL INVENTARIO
# ============================================================

st.header("🚦 2. Estado general del inventario")

status_summary = (
    filtered_df
    .groupby(status_col, as_index=False)
    .size()
    .rename(columns={"size": "registros"})
)

fig_status = px.pie(
    status_summary,
    names=status_col,
    values="registros",
    title="Distribución de estados del inventario",
    hole=0.45
)

fig_status.update_layout(height=430)

st.plotly_chart(fig_status, use_container_width=True)

st.info(
    """
    **Guía de lectura:**  
    Este gráfico muestra qué parte del inventario está en estado saludable, riesgo, desabasto o exceso.

    - Si crece la parte de **desabasto**, el negocio puede perder ventas.
    - Si crece la parte de **exceso**, puede haber dinero atrapado en almacén.
    - Lo ideal es que la mayor parte esté en **saludable**.
    """
)


# ============================================================
# GRÁFICO 3: COBERTURA DE INVENTARIO
# ============================================================

st.header("🧮 3. Cobertura de inventario por producto")

coverage_product = (
    filtered_df
    .groupby(product_col, as_index=False)
    .agg({
        "cobertura_inventario": "mean",
        stock_col: "sum",
        demand_col: "sum"
    })
)

coverage_product = coverage_product.sort_values(
    by="cobertura_inventario",
    ascending=True
)

fig_coverage = px.bar(
    coverage_product,
    x=product_col,
    y="cobertura_inventario",
    title="Cobertura promedio de inventario",
    labels={
        product_col: "Producto",
        "cobertura_inventario": "Cobertura de inventario"
    }
)

fig_coverage.add_hline(
    y=1,
    line_dash="dash",
    annotation_text="Límite mínimo: 1.0x",
    annotation_position="top left"
)

fig_coverage.add_hline(
    y=2.5,
    line_dash="dash",
    annotation_text="Posible exceso: 2.5x",
    annotation_position="top right"
)

fig_coverage.update_layout(
    xaxis_tickangle=-35,
    height=480
)

st.plotly_chart(fig_coverage, use_container_width=True)

st.info(
    """
    **Guía de lectura:**  
    La cobertura responde una pregunta sencilla:  
    **¿cuántas veces el inventario actual cubre la demanda esperada?**

    - Menos de **1.0x**: el inventario no alcanza.
    - Entre **1.3x y 2.5x**: zona razonable.
    - Más de **2.5x**: puede existir exceso de inventario.

    La métrica más importante aquí es la altura de la barra.
    """
)


# ============================================================
# GRÁFICO 4: ANÁLISIS POR REGIÓN
# ============================================================

st.header("🗺️ 4. Inventario por región")

region_summary = (
    filtered_df
    .groupby(region_col, as_index=False)
    .agg({
        stock_col: "sum",
        demand_col: "sum",
        "brecha_stock_demanda": "sum",
        "cobertura_inventario": "mean"
    })
)

region_summary = region_summary.sort_values(
    by="brecha_stock_demanda",
    ascending=True
)

fig_region = px.bar(
    region_summary,
    x=region_col,
    y="brecha_stock_demanda",
    title="Brecha entre stock y demanda por región",
    labels={
        region_col: "Región",
        "brecha_stock_demanda": "Stock - Demanda"
    }
)

fig_region.add_hline(
    y=0,
    line_dash="dash",
    annotation_text="Punto de equilibrio",
    annotation_position="top left"
)

fig_region.update_layout(
    xaxis_tickangle=-25,
    height=450
)

st.plotly_chart(fig_region, use_container_width=True)

st.info(
    """
    **Guía de lectura:**  
    Este gráfico muestra si a una región le sobra o le falta inventario.

    - Barras por encima de cero: hay más stock que demanda.
    - Barras por debajo de cero: falta inventario.
    - La línea en cero representa el punto de equilibrio.

    La señal más importante son las barras negativas, porque indican posible desabasto.
    """
)


# ============================================================
# GRÁFICO 5: TENDENCIA EN EL TIEMPO, SI EXISTE FECHA
# ============================================================

if date_col is not None and filtered_df[date_col].notna().any():

    st.header("📈 5. Evolución en el tiempo")

    time_summary = (
        filtered_df
        .dropna(subset=[date_col])
        .groupby(date_col, as_index=False)
        .agg({
            stock_col: "sum",
            demand_col: "sum"
        })
        .sort_values(by=date_col)
    )

    time_chart = time_summary.melt(
        id_vars=[date_col],
        value_vars=[stock_col, demand_col],
        var_name="Métrica",
        value_name="Cantidad"
    )

    fig_time = px.line(
        time_chart,
        x=date_col,
        y="Cantidad",
        color="Métrica",
        markers=True,
        title="Evolución de stock y demanda en el tiempo",
        labels={
            date_col: "Fecha",
            "Cantidad": "Cantidad",
            "Métrica": "Indicador"
        }
    )

    fig_time.update_layout(height=480)

    st.plotly_chart(fig_time, use_container_width=True)

    st.info(
        """
        **Guía de lectura:**  
        Este gráfico ayuda a ver cómo cambian el inventario y la demanda con el tiempo.

        - Si la línea de demanda sube y el stock no sube, puede aparecer desabasto.
        - Si el stock sube mucho y la demanda baja, puede aparecer exceso.
        - Lo ideal es que el inventario acompañe razonablemente el comportamiento de la demanda.
        """
    )


# ============================================================
# TABLA DE ALERTAS PRIORITARIAS
# ============================================================

st.header("🚨 Alertas prioritarias")

alerts_df = filtered_df.copy()
alerts_df["prioridad"] = alerts_df[status_col].apply(status_priority)

alerts_df = alerts_df.sort_values(
    by=["prioridad", "brecha_stock_demanda"],
    ascending=[True, True]
)

columns_to_show = [
    product_col,
    region_col,
    stock_col,
    demand_col,
    "brecha_stock_demanda",
    "cobertura_inventario",
    status_col
]

if sales_col is not None:
    columns_to_show.insert(4, sales_col)

columns_to_show = [col for col in columns_to_show if col in alerts_df.columns]

critical_df = alerts_df[
    alerts_df[status_col].astype(str).isin(["Desabasto", "Riesgo", "Exceso"])
][columns_to_show]

if critical_df.empty:
    st.success(
        "No hay alertas críticas con los filtros actuales. "
        "El inventario se ve estable en esta selección."
    )
else:
    st.warning(
        "Estos son los registros que conviene revisar primero. "
        "Aparecen ordenados por nivel de urgencia."
    )

    st.dataframe(
        critical_df,
        use_container_width=True,
        hide_index=True
    )

    st.caption(
        "Tip: empieza revisando los productos en desabasto, después los de riesgo, "
        "y al final los excesos de inventario."
    )


# ============================================================
# RECOMENDACIONES AUTOMÁTICAS SIMPLES
# ============================================================

st.header("🧠 Recomendaciones rápidas")

recommendations = []

if desabasto_count > 0:
    recommendations.append(
        "Priorizar reposición de productos en desabasto para evitar ventas perdidas."
    )

if riesgo_count > 0:
    recommendations.append(
        "Monitorear productos en riesgo porque tienen poco margen de seguridad."
    )

if exceso_count > 0:
    recommendations.append(
        "Revisar productos con exceso para evitar capital inmovilizado o saturación de almacén."
    )

if coverage_avg < 1:
    recommendations.append(
        "La cobertura promedio está por debajo de 1.0x; el inventario no alcanza para cubrir la demanda."
    )

if coverage_avg > 2.5:
    recommendations.append(
        "La cobertura promedio es alta; puede existir inventario de más en la selección actual."
    )

if not recommendations:
    recommendations.append(
        "El inventario se ve razonablemente equilibrado con los filtros actuales."
    )

for rec in recommendations:
    st.write(f"✅ {rec}")


# ============================================================
# SECCIÓN EDUCATIVA FINAL
# ============================================================

with st.expander("📚 Explicación simple de las métricas usadas"):
    st.markdown(
        """
        **Inventario total**  
        Es la cantidad de producto disponible.

        **Demanda esperada**  
        Es lo que se espera vender o necesitar.

        **Brecha stock vs demanda**  
        Se calcula así:  
        `stock - demanda`

        - Si da positivo, sobra inventario.
        - Si da negativo, falta inventario.

        **Cobertura de inventario**  
        Se calcula así:  
        `stock / demanda`

        - 1.0x significa que el inventario apenas cubre la demanda.
        - Menos de 1.0x significa posible desabasto.
        - Mucho más de 2.5x puede indicar exceso.
        """
    )


# ============================================================
# DATOS CRUDOS
# ============================================================

with st.expander("📄 Ver datos originales"):
    st.caption(
        "Esta sección es solo para revisión técnica. "
        "La parte importante para negocio está en los gráficos y alertas anteriores."
    )

    st.dataframe(
        filtered_df,
        use_container_width=True,
        hide_index=True
    )


# ============================================================
# PIE DE PÁGINA
# ============================================================

st.divider()

st.caption(
    "Dashboard diseñado como guía visual para usuarios no técnicos. "
    "La finalidad es facilitar decisiones rápidas sobre inventario, demanda, desabasto y exceso de stock."
)
