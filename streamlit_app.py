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
        dtype = series.dtype
        if dtype in [pl.Int64, pl.Float64, pl.Int32, pl.Float32]: return "Métrica"
        if series.n_unique() < 50: return "Categoria"
        return "Texto"

# ==========================================
# MOTOR DE INSIGHTS (STORYTELLING)
# ==========================================
class InsightEngine:
    @staticmethod
    def generate_summary(df):
        insights = []
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Int64, pl.Float64]]
        if numeric_cols:
            col = numeric_cols[0]
            val = df[col].mean()
            insights.append(f"A média de {col} é {val:.2f}.")
            insights.append(f"O valor máximo encontrado em {col} foi {df[col].max()}.")
        return " | ".join(insights)

# ==========================================
# MOTOR DE ANOMALIAS
# ==========================================
class AnomalyDetector:
    @staticmethod
    def find_anomalies(df, features):
        data = df.select(features).to_pandas().fillna(0)
        model = IsolationForest(contamination=0.05, random_state=42)
        return model.fit_predict(data)

# ==========================================
# MOTOR DE DADOS (DUCKDB + POLARS)
# ==========================================
class DataEngine:
    def __init__(self):
        self.con = duckdb.connect(database=':memory:')

    def load_data(self, file_content, filename):
        df = pl.read_csv(file_content) if filename.endswith('.csv') else pl.read_parquet(file_content)
        self.con.register('df_view', df.to_pandas())
        return df

    def query(self, sql):
        return self.con.execute(sql).df()

    def get_semantic_schema(self, df):
        return {col: DataProfiler.identify_type(col, df[col]) for col in df.columns}

# ==========================================
# UI E COMPONENTES
# ==========================================
class UITheme:
    @staticmethod
    def apply_custom_css():
        st.markdown("""
        <style>
            .main { background-color: #f8f9fa; }
            .kpi-card { background: white; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }
        </style>
        """, unsafe_allow_html=True)

# ==========================================
# APLICAÇÃO PRINCIPAL
# ==========================================
class DataVizApp:
    def __init__(self):
        self.theme = UITheme()
        self.engine = DataEngine()
        self.insight = InsightEngine()

    def run(self):
        self.theme.apply_custom_css()
        
        with st.sidebar:
            st.title("📂 DataViz Pro V1.2")
            uploaded_file = st.file_uploader("Carregar dataset", type=["csv", "parquet"])
        
        if uploaded_file:
            df = self.engine.load_data(uploaded_file, uploaded_file.name)
            schema = self.engine.get_semantic_schema(df)
            
            tabs = st.tabs(["Dashboard", "Anomalias", "Copilot IA"])
            
            with tabs[0]:
                st.subheader("Dashboard Executivo")
                st.success(self.insight.generate_summary(df))
                
                cols = [c for c, t in schema.items() if t == "Métrica"]
                if cols:
                    fig = px.histogram(df.to_pandas(), x=cols[0])
                    st.plotly_chart(fig, use_container_width=True)

            with tabs[1]: # ANOMALIAS
                st.subheader("Detecção de Anomalias")
                num_cols = [c for c, t in schema.items() if t == "Métrica"]
                if len(num_cols) >= 2:
                    sel = st.multiselect("Features", num_cols, default=num_cols[:2])
                    if st.button("Detectar"):
                        preds = AnomalyDetector.find_anomalies(df, sel)
                        pdf = df.to_pandas()
                        pdf['Anomalia'] = ["Sim" if p == -1 else "Não" for p in preds]
                        st.dataframe(pdf[pdf['Anomalia'] == "Sim"])
            
            with tabs[2]: # COPILOT
                st.subheader("Copilot de Dados")
                query = st.text_input("Ex: 'SELECT * FROM df_view LIMIT 5'")
                if query:
                    try:
                        res = self.engine.query(query)
                        st.dataframe(res)
                    except Exception as e:
                        st.error(f"Erro SQL: {e}")

        else:
            st.info("Carregue um arquivo para ativar o motor DuckDB e Análises.")

if __name__ == "__main__":
    app = DataVizApp()
    app.run()
