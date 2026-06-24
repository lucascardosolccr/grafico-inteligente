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
st.set_page_config(page_title="DataViz Pro Engine", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# MOTOR DE PERFILAMENTO SEMÂNTICO
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
# MOTOR DE TRANSFORMAÇÃO (LIMPEZA)
# ==========================================
class DataTransformer:
    @staticmethod
    def remove_duplicates(df):
        return df.unique()

    @staticmethod
    def fill_nulls(df):
        return df.fill_null(strategy="forward")

# ==========================================
# MOTOR DE INSIGHTS
# ==========================================
class InsightEngine:
    @staticmethod
    def generate_summary(df):
        insights = []
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Int64, pl.Float64]]
        if numeric_cols:
            col = numeric_cols[0]
            val = df[col].mean()
            insights.append(f"A média aritmética de **{col}** é {val:.2f}.")
            insights.append(f"O valor máximo registrado em **{col}** é {df[col].max()}.")
        return " | ".join(insights)

# ==========================================
# MOTOR DE ANOMALIAS
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
    def __init__(self):
        self.con = duckdb.connect(database=':memory:')

    @st.cache_data
    def load_data(_self, file_content, filename):
        try:
            ext = filename.split('.')[-1].lower()
            if ext == 'csv': df = pl.read_csv(file_content)
            elif ext in ['xlsx', 'xls']: df = pl.from_pandas(pd.read_excel(file_content))
            elif ext == 'parquet': df = pl.read_parquet(file_content)
            elif ext == 'json': df = pl.read_json(file_content)
            else: return None
            _self.con.register('df_view', df.to_pandas())
            return df
        except Exception as e:
            st.error(f"Erro no processamento: {e}")
            return None

    def query(self, sql):
        return self.con.execute(sql).df()

    def get_semantic_schema(self, df):
        return {col: DataProfiler.identify_type(col, df[col]) for col in df.columns}

# ==========================================
# MOTOR DE VISUALIZAÇÃO
# ==========================================
class GraphEngine:
    @staticmethod
    def create_plot(df, plot_type, x, y, color=None):
        pdf = df.to_pandas()
        # Validação defensiva para hierarquias
        val_y = y if pd.api.types.is_numeric_dtype(pdf[y]) else None
        
        if plot_type == "Linha": return px.line(pdf, x=x, y=y)
        elif plot_type == "Barras": return px.bar(pdf, x=x, y=y, color=color)
        elif plot_type == "Dispersão": return px.scatter(pdf, x=x, y=y, color=color)
        elif plot_type == "Histograma": return px.histogram(pdf, x=x)
        elif plot_type == "Treemap": return px.treemap(pdf, path=[x], values=val_y)
        elif plot_type == "Sunburst": return px.sunburst(pdf, path=[x], values=val_y)
        return px.bar(pdf, x=x, y=y)

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
# MOTOR DE ML
# ==========================================
class MLProcessor:
    @staticmethod
    def run_kmeans(df, features, n_clusters):
        data = df.select(features).to_pandas().select_dtypes(include=[np.number]).dropna()
        if data.empty: return None
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(scaled_data)

# ==========================================
# UI E ESTILIZAÇÃO
# ==========================================
class UITheme:
    @staticmethod
    def apply_custom_css():
        st.markdown("""
        <style>
            .main { background-color: #f8f9fa; }
            .kpi-card { background: white; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; }
            .kpi-value { font-size: 20px; font-weight: bold; color: #2c3e50; }
            .kpi-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_kpi(label, value):
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value">{value}</div>
        </div>
        """, unsafe_allow_html=True)

# ==========================================
# APLICAÇÃO PRINCIPAL
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
            st.title("📂 DataViz Pro V2.0")
            uploaded_file = st.file_uploader("Carregar dataset", type=["csv", "xlsx", "json", "parquet"])
        
        st.title("📊 Painel de Análise Profissional")
        if uploaded_file:
            df = self.engine.load_data(uploaded_file, uploaded_file.name)
            if df is not None:
                schema = self.engine.get_semantic_schema(df)
                tabs = st.tabs(["Dashboard", "Exploratória", "Estúdio Visual", "Transformação", "Clusterização", "Anomalias", "Assistente IA"])
                
                with tabs[0]: # DASHBOARD
                    st.subheader("Dashboard Executivo")
                    st.metric("Qualidade dos Dados", f"{DataProfiler.calculate_quality_score(df)}/100")
                    st.info(self.insight.generate_summary(df))
                    cols = [c for c, t in schema.items() if t == "Métrica"]
                    if cols:
                        cols_ui = st.columns(min(len(cols), 4))
                        for i, m in enumerate(cols[:4]):
                            with cols_ui[i]: self.theme.render_kpi(m, f"{df[m].sum():,.0f}")
                    st.plotly_chart(self.viz.auto_plot(df, schema), use_container_width=True, key="dash_chart")

                with tabs[1]: # EXPLORATÓRIA
                    st.subheader("Configuração Manual de Gráficos")
                    c1, c2 = st.columns(2)
                    x = c1.selectbox("Eixo X (Manual)", list(df.columns), key="expl_x")
                    y = c2.selectbox("Eixo Y (Manual)", list(df.columns), key="expl_y")
                    st.plotly_chart(self.viz.create_plot(df, "Barras", x, y), use_container_width=True, key="expl_chart")

                with tabs[2]: # ESTÚDIO VISUAL
                    st.subheader("Estúdio de Visualização")
                    c1, c2, c3 = st.columns([1, 3, 1])
                    with c1:
                        gx = st.selectbox("Eixo X (Estúdio)", list(df.columns), key="studio_x")
                        gy = st.selectbox("Eixo Y (Estúdio)", list(df.columns), key="studio_y")
                        gtype = st.selectbox("Tipo de Gráfico", ["Barras", "Linha", "Dispersão", "Histograma", "Treemap", "Sunburst"], key="studio_type")
                    with c2: st.plotly_chart(self.viz.create_plot(df, gtype, gx, gy), use_container_width=True, key="studio_chart")

                with tabs[3]: # TRANSFORMAÇÃO
                    st.subheader("Engenharia de Dados")
                    if st.button("Remover Duplicados", key="dup_btn"): df = self.transformer.remove_duplicates(df)
                    if st.button("Preencher Nulos", key="null_btn"): df = self.transformer.fill_nulls(df)
                    st.dataframe(df.to_pandas().head())

                with tabs[4]: # ML CLUSTER
                    num_cols = [c for c, t in schema.items() if t == "Métrica"]
                    sel = st.multiselect("Features", num_cols, default=num_cols[:2], key="ml_sel")
                    k = st.slider("Clusters", 2, 6, 3, key="ml_k")
                    if st.button("Executar Clusterização", key="ml_btn"):
                        res = self.ml.run_kmeans(df, sel, k)
                        if res is not None: st.write("Clusterização finalizada.")

                with tabs[5]: # ANOMALIAS
                    st.subheader("Detecção de Anomalias")
                    num_cols = [c for c, t in schema.items() if t == "Métrica"]
                    sel = st.multiselect("Features Anomalia", num_cols, default=num_cols[:2], key="ano_sel")
                    if st.button("Rodar Detecção", key="ano_btn"):
                        preds = AnomalyDetector.find_anomalies(df, sel)
                        pdf = df.to_pandas()
                        pdf['Status'] = ["Anômalo" if p == -1 else "Normal" for p in preds]
                        st.dataframe(pdf[pdf['Status'] == "Anômalo"])

                with tabs[6]: # ASSISTENTE IA
                    query = st.text_input("Consulta SQL:", key="sql_input")
                    if query: st.dataframe(self.engine.query(query))
                    if prompt := st.chat_input("Pergunte algo...", key="ai_chat"):
                        st.chat_message("user").markdown(prompt)
                        st.chat_message("assistant").markdown("Motor de IA ativo.")
        else:
            st.info("Carregue um arquivo para iniciar.")

if __name__ == "__main__":
    app = DataVizApp()
    app.run()
