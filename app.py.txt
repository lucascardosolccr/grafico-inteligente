import streamlit as st
import polars as pl
import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ==========================================
# CONFIGURAÇÃO GLOBAL
# ==========================================
st.set_page_config(page_title="DataViz Pro Engine", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# CLASSE DE ESTILIZAÇÃO E UI
# ==========================================
class UITheme:
    @staticmethod
    def apply_custom_css():
        st.markdown("""
        <style>
            .main { background-color: #f5f7f9; }
            section[data-testid="stSidebar"] { background-color: #ffffff; border-right: 1px solid #e0e0e0; }
            .kpi-card { background: white; padding: 20px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }
            .kpi-value { font-size: 24px; font-weight: bold; color: #007bff; }
            .kpi-label { font-size: 14px; color: #555; }
            h1, h2, h3 { color: #2c3e50; }
            div.stButton > button { width: 100%; border-radius: 5px; border: 1px solid #007bff; color: white; background-color: #007bff; }
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
    @st.cache_data
    def run_kmeans(data_bytes, features, n_clusters):
        # Recria df para processamento interno
        df = pl.read_parquet(data_bytes) if isinstance(data_bytes, bytes) else data_bytes
        data = df.select(features).to_pandas()
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
        """Carrega e otimiza o dataset com cache."""
        try:
            ext = filename.split('.')[-1].lower()
            if ext == 'csv': return pl.read_csv(file_content)
            elif ext in ['xlsx', 'xls']: return pl.from_pandas(pd.read_excel(file_content))
            elif ext == 'parquet': return pl.read_parquet(file_content)
            elif ext == 'json': return pl.read_json(file_content)
            return None
        except Exception as e:
            st.error(f"Erro no processamento: {e}")
            return None

    @staticmethod
    def get_column_info(df):
        info = {}
        for col in df.columns:
            dtype = df[col].dtype
            info[col] = "Numérico" if dtype in [pl.Int64, pl.Float64, pl.Int32] else "Categórico"
        return info

# ==========================================
# MOTOR DE VISUALIZAÇÃO
# ==========================================
class GraphEngine:
    @staticmethod
    def create_plot(df, plot_type, x, y):
        pdf = df.to_pandas()
        if plot_type == "Linha": return px.line(pdf, x=x, y=y)
        elif plot_type == "Barras": return px.bar(pdf, x=x, y=y)
        elif plot_type == "Dispersão": return px.scatter(pdf, x=x, y=y)
        return px.bar(pdf, x=x, y=y)

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
                tabs = st.tabs(["Dashboard", "Análise Exploratória", "Clusterização ML", "Assistente IA"])
                
                with tabs[0]: # DASHBOARD
                    st.subheader("Indicadores Chave")
                    cols = [c for c, t in self.engine.get_column_info(df).items() if t == "Numérico"]
                    if cols:
                        stats = {"Soma": df[cols[0]].sum(), "Média": round(df[cols[0]].mean(), 2)}
                        c1, c2 = st.columns(2)
                        with c1: self.theme.render_kpi("Soma Total", stats['Soma'])
                        with c2: self.theme.render_kpi("Média", stats['Média'])
                
                with tabs[1]: # EXPLORATÓRIA
                    col_info = self.engine.get_column_info(df)
                    c1, c2 = st.columns(2)
                    x = c1.selectbox("Eixo X", list(col_info.keys()))
                    y = c2.selectbox("Eixo Y", list(col_info.keys()))
                    fig = self.viz.create_plot(df, "Barras", x, y)
                    st.plotly_chart(fig, use_container_width=True)
                
                with tabs[2]: # ML
                    st.subheader("Machine Learning: Clusterização")
                    cols = [c for c, t in self.engine.get_column_info(df).items() if t == "Numérico"]
                    if len(cols) >= 2:
                        sel_cols = st.multiselect("Selecione colunas", cols, default=cols[:2])
                        k = st.slider("Clusters", 2, 6, 3)
                        if st.button("Executar Clusterização"):
                            clusters = self.ml.run_kmeans(df, sel_cols, k)
                            pdf = df.to_pandas()
                            pdf['Cluster'] = clusters
                            st.scatter_chart(pdf, x=sel_cols[0], y=sel_cols[1], color='Cluster')
                
                with tabs[3]: # ASSISTENTE
                    st.subheader("Assistente de Dados")
                    if prompt := st.chat_input("Pergunte algo sobre seus dados..."):
                        with st.chat_message("user"): st.markdown(prompt)
                        with st.chat_message("assistant"): 
                            st.markdown(f"Analisando: *'{prompt}'*... Identifiquei {len(df.columns)} colunas e {len(df)} registros.")

        else:
            st.info("Por favor, carregue um arquivo na barra lateral para começar.")

if __name__ == "__main__":
    app = DataVizApp()
    app.run()