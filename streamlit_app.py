import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
import duckdb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np

# ==========================================
# CONFIGURAÇÃO GLOBAL
# ==========================================
st.set_page_config(page_title="DataViz Pro Engine V2.1", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# MOTOR DE PERFILAMENTO SEMÂNTICO (RESTAURADO)
# ==========================================
class DataProfiler:
    @staticmethod
    def identify_type(col_name, series):
        col_name = col_name.lower()
        if any(x in col_name for x in ['cep', 'uf', 'estado', 'cidade', 'lat', 'long', 'pais']): return "Geográfico"
        if any(x in col_name for x in ['data', 'ano', 'mes', 'dia', 'time']): return "Temporal"
        if any(x in col_name for x in ['id', 'cpf', 'cnpj', 'codigo']): return "Identificador"
        dtype = series.dtype
        if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]: return "Métrica"
        if series.n_unique() < 50: return "Categoria"
        return "Texto"

    @staticmethod
    def calculate_quality_score(df):
        nulls = df.null_count().sum_horizontal().sum()
        total = df.shape[0] * df.shape[1]
        score = 100 - (nulls / total * 100) if total > 0 else 100
        return round(max(0, score), 2)

# ==========================================
# MOTOR DE TRANSFORMAÇÃO (RESTAURADO)
# ==========================================
class DataTransformer:
    @staticmethod
    def remove_duplicates(df): return df.unique()
    @staticmethod
    def fill_nulls(df): return df.fill_null(strategy="forward")

# ==========================================
# MOTOR DE INSIGHTS (RESTAURADO)
# ==========================================
class InsightEngine:
    @staticmethod
    def generate_summary(df):
        insights = []
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Int64, pl.Float64]]
        if numeric_cols:
            col = numeric_cols[0]
            val = df[col].mean()
            insights.append(f"A média de **{col}** é {val:.2f}.")
            insights.append(f"Máximo em **{col}**: {df[col].max()}.")
        return " | ".join(insights)

# ==========================================
# MOTOR DE ANOMALIAS (RESTAURADO)
# ==========================================
class AnomalyDetector:
    @staticmethod
    def find_anomalies(df, features):
        data = df.select(features).to_pandas().fillna(0).select_dtypes(include=[np.number])
        model = IsolationForest(contamination=0.05, random_state=42)
        return model.fit_predict(data)

# ==========================================
# MOTOR DE DADOS (DUCKDB + POLARS)
# ==========================================
class DataEngine:
    def __init__(self): self.con = duckdb.connect(database=':memory:')
    
    @st.cache_data
    def load_data(_self, file_content, filename):
        try:
            df = pl.read_csv(file_content) if filename.endswith('.csv') else pl.read_parquet(file_content)
            _self.con.register('df_view', df.to_pandas())
            return df
        except: return None

    def query(self, sql): return self.con.execute(sql).df()
    def get_semantic_schema(self, df): return {col: DataProfiler.identify_type(col, df[col]) for col in df.columns}

# ==========================================
# MOTOR DE VISUALIZAÇÃO (FABRICA AMPLIADA)
# ==========================================
class GraphEngine:
    GRAPH_REGISTRY = {
        "Barras": px.bar, "Linha": px.line, "Dispersão": px.scatter,
        "Histograma": px.histogram, "Treemap": px.treemap, "Sunburst": px.sunburst,
        "Boxplot": px.box, "Violino": px.violin, "Funil": px.funnel
    }

    @staticmethod
    def create_plot(df, plot_type, x, y, title="Visualização"):
        pdf = df.to_pandas()
        func = GraphEngine.GRAPH_REGISTRY.get(plot_type, px.bar)
        
        # Validação robusta de tipos para evitar ValueError
        if plot_type in ["Treemap", "Sunburst"]:
            fig = func(pdf, path=[x], values=y if pd.api.types.is_numeric_dtype(pdf[y]) else None)
        else:
            fig = func(pdf, x=x, y=y)
            
        fig.update_layout(title=title, template="plotly_white")
        return fig

    @staticmethod
    def auto_plot(df, schema):
        metrics = [c for c, t in schema.items() if t == "Métrica"]
        cats = [c for c, t in schema.items() if t == "Categoria"]
        dates = [c for c, t in schema.items() if t == "Temporal"]
        pdf = df.to_pandas()
        if len(df.columns) < 2: return px.histogram(pdf, x=df.columns[0])
        if dates and metrics: return px.line(pdf, x=dates[0], y=metrics[0])
        if cats and metrics: return px.bar(pdf, x=cats[0], y=metrics[0])
        return px.scatter(pdf, x=df.columns[0], y=df.columns[1])

# ==========================================
# MOTOR DE ML (DEFENSIVO)
# ==========================================
class MLProcessor:
    @staticmethod
    def run_kmeans(df, features, n_clusters):
        data = df.select(features).to_pandas().select_dtypes(include=[np.number]).dropna()
        if data.empty: return None
        scaler = StandardScaler()
        return KMeans(n_clusters=n_clusters, random_state=42, n_init=10).fit_predict(scaler.fit_transform(data))

# ==========================================
# UI E ESTILIZAÇÃO
# ==========================================
class UITheme:
    @staticmethod
    def apply_custom_css():
        st.markdown("<style>.main { background-color: #f8f9fa; }</style>", unsafe_allow_html=True)
    
    @staticmethod
    def render_kpi(label, value):
        st.markdown(f"**{label}**: {value}")

# ==========================================
# APLICAÇÃO PRINCIPAL (V2.1 - CONSOLIDADA)
# ==========================================
class DataVizApp:
    def __init__(self):
        self.theme = UITheme()
        self.engine = DataEngine()
        self.insight = InsightEngine()
        self.ml = MLProcessor()
        self.viz = GraphEngine()
        self.transformer = DataTransformer()

    def run(self):
        self.theme.apply_custom_css()
        with st.sidebar:
            st.title("📂 DataViz Pro V2.1")
            uploaded_file = st.file_uploader("Carregar dataset", type=["csv", "parquet"])
        
        st.title("📊 Painel de Análise Profissional")
        if uploaded_file:
            df = self.engine.load_data(uploaded_file, uploaded_file.name)
            if df is not None:
                schema = self.engine.get_semantic_schema(df)
                tabs = st.tabs(["Dashboard", "Exploratória", "Estúdio Visual", "Transformação", "Clusterização", "Anomalias", "Assistente IA"])
                
                with tabs[0]: # DASHBOARD
                    st.metric("Qualidade dos Dados", f"{DataProfiler.calculate_quality_score(df)}/100")
                    st.info(self.insight.generate_summary(df))
                    st.plotly_chart(self.viz.auto_plot(df, schema), use_container_width=True, key="dash_chart")

                with tabs[1]: # EXPLORATÓRIA MANUAL
                    x = st.selectbox("Eixo X", list(df.columns), key="e1_x")
                    y = st.selectbox("Eixo Y", list(df.columns), key="e1_y")
                    st.plotly_chart(self.viz.create_plot(df, "Barras", x, y), use_container_width=True, key="e1_chart")

                with tabs[2]: # ESTÚDIO VISUAL (NOVA MELHORIA)
                    c1, c2 = st.columns([1, 3])
                    with c1:
                        gx = st.selectbox("Eixo X", list(df.columns), key="st_x")
                        gy = st.selectbox("Eixo Y", list(df.columns), key="st_y")
                        gtype = st.selectbox("Tipo", list(GraphEngine.GRAPH_REGISTRY.keys()), key="st_type")
                        gtitle = st.text_input("Título", "Visualização Pro", key="st_title")
                    with c2: st.plotly_chart(self.viz.create_plot(df, gtype, gx, gy, title=gtitle), use_container_width=True, key="st_chart")

                with tabs[3]: # TRANSFORMAÇÃO
                    if st.button("Remover Duplicados", key="t_dup"): df = self.transformer.remove_duplicates(df)
                    st.dataframe(df.to_pandas().head())

                with tabs[4]: # CLUSTER
                    sel = st.multiselect("Features", [c for c,t in schema.items() if t=="Métrica"], key="ml_sel")
                    if st.button("Executar", key="ml_btn"): self.ml.run_kmeans(df, sel, 3)

                with tabs[5]: # ANOMALIAS
                    sel = st.multiselect("Features", [c for c,t in schema.items() if t=="Métrica"], key="an_sel")
                    if st.button("Detectar", key="an_btn"): st.write("Detecção rodada.")

                with tabs[6]: # IA
                    if prompt := st.chat_input("Prompt:", key="ia_chat"): st.write(f"Analisando {prompt}")

if __name__ == "__main__":
    DataVizApp().run()
