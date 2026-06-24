import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
import re

# ==========================================
# CONFIGURAÇÃO GLOBAL
# ==========================================
st.set_page_config(page_title="DataViz Pro Engine", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# MOTOR DE PERFILAMENTO SEMÂNTICO (O CÉREBRO)
# ==========================================
class DataProfiler:
    @staticmethod
    def identify_type(col_name, series):
        """Identifica semântica da coluna."""
        col_name = col_name.lower()
        
        # Geográfico
        if any(x in col_name for x in ['cep', 'uf', 'estado', 'cidade', 'lat', 'long', 'pais']):
            return "Geográfico"
        # Temporal
        if any(x in col_name for x in ['data', 'ano', 'mes', 'dia', 'time']):
            return "Temporal"
        # Identificadores
        if any(x in col_name for x in ['id', 'cpf', 'cnpj', 'codigo']):
            return "Identificador"
        
        # Tipagem nativa
        dtype = series.dtype
        if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]:
            return "Métrica"
        
        # Categorias (Cardinalidade baixa)
        if series.n_unique() < 50:
            return "Categoria"
            
        return "Texto"

# ==========================================
# CLASSE DE ESTILIZAÇÃO E UI
# ==========================================
class UITheme:
    @staticmethod
    def apply_custom_css():
        st.markdown("""
        <style>
            .main { background-color: #f8f9fa; }
            .kpi-card { background: white; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; margin-bottom: 10px;}
            .kpi-value { font-size: 20px; font-weight: bold; color: #2c3e50; }
            .kpi-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; }
            div.stButton > button { width: 100%; border-radius: 5px; }
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
# MOTOR DE MACHINE LEARNING
# ==========================================
class MLProcessor:
    @staticmethod
    def run_kmeans(df, features, n_clusters):
        data = df.select(features).to_pandas().dropna()
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(scaled_data)

# ==========================================
# MOTOR DE DADOS
# ==========================================
class DataEngine:
    @staticmethod
    @st.cache_data
    def load_data(file_content, filename):
        try:
            ext = filename.split('.')[-1].lower()
            if ext == 'csv': return pl.read_csv(file_content)
            elif ext in ['xlsx', 'xls']: return pl.from_pandas(pd.read_excel(file_content))
            elif ext == 'parquet': return pl.read_parquet(file_content)
            elif ext == 'json': return pl.read_json(file_content)
            return None
        except Exception as e:
            st.error(f"Erro: {e}")
            return None

    def get_semantic_schema(self, df):
        schema = {}
        for col in df.columns:
            schema[col] = DataProfiler.identify_type(col, df[col])
        return schema

# ==========================================
# MOTOR DE VISUALIZAÇÃO
# ==========================================
class GraphEngine:
    @staticmethod
    def auto_plot(df, schema):
        """Sugere e gera o melhor gráfico baseado no schema semântico."""
        metrics = [c for c, t in schema.items() if t == "Métrica"]
        cats = [c for c, t in schema.items() if t == "Categoria"]
        dates = [c for c, t in schema.items() if t == "Temporal"]
        
        pdf = df.to_pandas()
        
        # 1. Temporal + Métrica
        if dates and metrics:
            return px.line(pdf, x=dates[0], y=metrics[0], title=f"Evolução de {metrics[0]}")
        # 2. Categoria + Métrica
        if cats and metrics:
            return px.bar(pdf, x=cats[0], y=metrics[0], title=f"{metrics[0]} por {cats[0]}")
        # 3. Dispersão se tiver duas métricas
        if len(metrics) >= 2:
            return px.scatter(pdf, x=metrics[0], y=metrics[1], title=f"Correlação {metrics[0]} vs {metrics[1]}")
        
        return px.bar(pdf, x=df.columns[0], y=df.columns[1])

# ==========================================
# APLICAÇÃO PRINCIPAL
# ==========================================
class DataVizApp:
    def __init__(self):
        self.theme = UITheme()
        self.engine = DataEngine()
        self.ml = MLProcessor()
        self.viz = GraphEngine()

    def run(self):
        self.theme.apply_custom_css()
        
        with st.sidebar:
            st.title("📂 DataViz Pro")
            uploaded_file = st.file_uploader("Carregar dataset", type=["csv", "xlsx", "json", "parquet"])
        
        st.title("📊 Painel de Análise Profissional")
        
        if uploaded_file:
            df = self.engine.load_data(uploaded_file, uploaded_file.name)
            if df is not None:
                schema = self.engine.get_semantic_schema(df)
                
                tabs = st.tabs(["Dashboard Automático", "Análise Exploratória", "Clusterização ML", "Assistente IA"])
                
                with tabs[0]: # DASHBOARD
                    st.subheader("Insights Gerados Automaticamente")
                    metrics = [c for c, t in schema.items() if t == "Métrica"]
                    
                    # KPIs
                    if metrics:
                        cols_ui = st.columns(min(len(metrics), 4))
                        for i, m in enumerate(metrics[:4]):
                            with cols_ui[i]:
                                self.theme.render_kpi(m, f"{df[m].sum():,.2f}")
                    
                    # Gráficos sugeridos
                    fig = self.viz.auto_plot(df, schema)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tabs[1]: # EXPLORATÓRIA
                    st.write("Configuração Manual de Gráficos")
                    c1, c2 = st.columns(2)
                    x = c1.selectbox("Eixo X", list(df.columns))
                    y = c2.selectbox("Eixo Y", list(df.columns))
                    st.plotly_chart(self.viz.auto_plot(df, {x:"Categoria", y:"Métrica"}), use_container_width=True)
                
                with tabs[2]: # ML
                    st.subheader("Machine Learning")
                    cols = [c for c, t in schema.items() if t == "Métrica"]
                    if len(cols) >= 2:
                        sel_cols = st.multiselect("Selecione métricas", cols, default=cols[:2])
                        if st.button("Executar Clusterização"):
                            clusters = self.ml.run_kmeans(df, sel_cols, 3)
                            pdf = df.to_pandas()
                            pdf['Cluster'] = clusters
                            st.scatter_chart(pdf, x=sel_cols[0], y=sel_cols[1], color='Cluster')
                
                with tabs[3]: # ASSISTENTE
                    st.subheader("Assistente de Dados (Beta)")
                    if prompt := st.chat_input("Ex: 'Qual a soma de receita por categoria?'"):
                        st.chat_message("user").markdown(prompt)
                        st.chat_message("assistant").markdown("Em desenvolvimento: Motor SQL/Query de dados conectado ao DataProfiler.")

        else:
            st.info("Carregue um arquivo para iniciar o Motor Semântico.")

if __name__ == "__main__":
    app = DataVizApp()
    app.run()
