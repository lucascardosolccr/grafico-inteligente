import streamlit as st
import streamlit.components.v1 as components 
import polars as pl
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import duckdb
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import IsolationForest
import numpy as np
import sqlite3
import json
import uuid
import copy

# ==========================================
# NOVOS IMPORTS - VOLUME 01
# ==========================================
import pyarrow as pa

try: import vaex
except ImportError: vaex = None

try: import ydata_profiling
except ImportError: ydata_profiling = None

try: import sweetviz as sv
except ImportError: sv = None

try: from autoviz.AutoViz_Class import AutoViz_Class
except ImportError: AutoViz_Class = None

try: from streamlit_echarts import st_echarts
except ImportError: st_echarts = None

try: import altair as alt
except ImportError: alt = None

try: import pygwalker as pyg
except ImportError: pyg = None

try: from streamlit_elements import elements, mui, html, dashboard
except ImportError: elements = mui = html = dashboard = None

# ==========================================
# CONFIGURAÇÃO GLOBAL
# ==========================================
st.set_page_config(page_title="DataViz Pro Engine", layout="wide", initial_sidebar_state="expanded")

# ==========================================
# MOTOR DE BANCO DE DADOS (VOLUME 02)
# ==========================================
class ProjectRepository:
    def __init__(self):
        self.conn = sqlite3.connect("projects.db", check_same_thread=False)
        self._init_schema()

    def _init_schema(self):
        cursor = self.conn.cursor()
        cursor.execute("CREATE TABLE IF NOT EXISTS projects (id TEXT PRIMARY KEY, name TEXT, updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)")
        cursor.execute("CREATE TABLE IF NOT EXISTS dashboards (id TEXT PRIMARY KEY, project_id TEXT, name TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS charts (id TEXT PRIMARY KEY, dashboard_id TEXT, type TEXT, config TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS layers (id TEXT PRIMARY KEY, dashboard_id TEXT, z_index INTEGER, payload TEXT)")
        cursor.execute("CREATE TABLE IF NOT EXISTS themes (id TEXT PRIMARY KEY, name TEXT, config TEXT)")
        self.conn.commit()

    def save_canvas_state(self, project_name, canvas_objects):
        cursor = self.conn.cursor()
        p_id = str(uuid.uuid4())
        d_id = str(uuid.uuid4())
        
        cursor.execute("INSERT INTO projects (id, name) VALUES (?, ?)", (p_id, project_name))
        cursor.execute("INSERT INTO dashboards (id, project_id, name) VALUES (?, ?, ?)", (d_id, p_id, "Principal"))
        
        for obj in canvas_objects:
            cursor.execute(
                "INSERT INTO layers (id, dashboard_id, z_index, payload) VALUES (?, ?, ?, ?)",
                (obj['id'], d_id, 1, json.dumps(obj))
            )
        self.conn.commit()
        return p_id

    def load_projects(self):
        return pd.read_sql_query("SELECT id, name, updated_at FROM projects ORDER BY updated_at DESC", self.conn)

    def load_canvas_state(self, project_id):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id FROM dashboards WHERE project_id = ? LIMIT 1", (project_id,))
        dash_row = cursor.fetchone()
        if not dash_row: return []
        
        cursor.execute("SELECT payload FROM layers WHERE dashboard_id = ?", (dash_row[0],))
        layers = cursor.fetchall()
        return [json.loads(row[0]) for row in layers]

# ==========================================
# MOTOR DE HISTÓRICO (UNDO/REDO - VOL 02)
# ==========================================
class HistoryManager:
    @staticmethod
    def init_state():
        if 'history' not in st.session_state:
            st.session_state['history'] = [[]] 
            st.session_state['history_index'] = 0
            st.session_state['canvas_objects'] = []

    @staticmethod
    def push_state(new_objects):
        current_idx = st.session_state['history_index']
        st.session_state['history'] = st.session_state['history'][:current_idx + 1]
        st.session_state['history'].append(copy.deepcopy(new_objects))
        st.session_state['history_index'] += 1
        st.session_state['canvas_objects'] = copy.deepcopy(new_objects)

    @staticmethod
    def undo():
        if st.session_state['history_index'] > 0:
            st.session_state['history_index'] -= 1
            st.session_state['canvas_objects'] = copy.deepcopy(st.session_state['history'][st.session_state['history_index']])

    @staticmethod
    def redo():
        if st.session_state['history_index'] < len(st.session_state['history']) - 1:
            st.session_state['history_index'] += 1
            st.session_state['canvas_objects'] = copy.deepcopy(st.session_state['history'][st.session_state['history_index']])

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

class DataTransformer:
    @staticmethod
    def remove_duplicates(df): return df.unique()
    @staticmethod
    def fill_nulls(df): return df.fill_null(strategy="forward")

class InsightEngine:
    @staticmethod
    def generate_summary(df):
        insights = []
        numeric_cols = [c for c in df.columns if df[c].dtype in [pl.Int64, pl.Float64]]
        if numeric_cols:
            col = numeric_cols[0]
            val = df[col].mean()
            insights.append(f"A média aritmética de **{col}** é {val:.2f}.")
            insights.append(f"O valor máximo de **{col}** é {df[col].max()}.")
        return " | ".join(insights)

class AnomalyDetector:
    @staticmethod
    def find_anomalies(df, features):
        data = df.select(features).to_pandas().fillna(0).select_dtypes(include=[np.number])
        model = IsolationForest(contamination=0.05, random_state=42)
        return model.fit_predict(data)

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

    def query(self, sql): return self.con.execute(sql).df()
    def get_semantic_schema(self, df): return {col: DataProfiler.identify_type(col, df[col]) for col in df.columns}

# ==========================================
# MOTOR DE VISUALIZAÇÃO (ATUALIZADO - VOL 02)
# ==========================================
class GraphEngine:
    @staticmethod
    def create_plot(df, plot_type, x, y, color=None, config=None):
        pdf = df.to_pandas()
        val_y = y if pd.api.types.is_numeric_dtype(pdf[y]) else None
        
        if plot_type == "Linha": fig = px.line(pdf, x=x, y=y)
        elif plot_type == "Barras": fig = px.bar(pdf, x=x, y=y, color=color)
        elif plot_type == "Dispersão": fig = px.scatter(pdf, x=x, y=y, color=color)
        elif plot_type == "Histograma": fig = px.histogram(pdf, x=x)
        elif plot_type == "Treemap": fig = px.treemap(pdf, path=[x], values=val_y)
        elif plot_type == "Sunburst": fig = px.sunburst(pdf, path=[x], values=val_y)
        elif plot_type == "Area": fig = px.area(pdf, x=x, y=y)
        elif plot_type == "Radar": fig = px.line_polar(pdf, r=y, theta=x, line_close=True) if val_y else px.bar(pdf, x=x, y=y, title="⚠️ Eixo Y inválido")
        elif plot_type == "Polar": fig = px.scatter_polar(pdf, r=y, theta=x) if val_y else px.scatter(pdf, x=x, y=y, title="⚠️ Eixo Y inválido")
        elif plot_type == "Funnel": fig = px.funnel(pdf, x=x, y=y)
        elif plot_type == "Waterfall": fig = go.Figure(go.Waterfall(x=pdf[x], y=pdf[y]))
        elif plot_type == "Violin": fig = px.violin(pdf, x=x, y=y)
        elif plot_type == "Boxplot": fig = px.box(pdf, x=x, y=y)
        elif plot_type == "Heatmap": fig = px.density_heatmap(pdf, x=x, y=y)
        elif plot_type == "Density Map": fig = px.density_mapbox(pdf, lat=x, lon=y, radius=10, mapbox_style="carto-positron") if 'lat' in x.lower() else px.density_heatmap(pdf, x=x, y=y)
        else: fig = px.bar(pdf, x=x, y=y)

        if config and hasattr(fig, 'update_layout'):
            fig.update_layout(
                title=config.get('title', ''),
                plot_bgcolor=config.get('plot_bgcolor', None),
                paper_bgcolor=config.get('paper_bgcolor', None)
            )
            
            x_params = {}
            if config.get('invert_x'): x_params['autorange'] = 'reversed'
            if config.get('scale_x') == 'Log': x_params['type'] = 'log'
            if x_params: fig.update_xaxes(**x_params)

            y_params = {}
            if config.get('invert_y'): y_params['autorange'] = 'reversed'
            if config.get('scale_y') == 'Log': y_params['type'] = 'log'
            fmt = config.get('format_y', 'Linear')
            if fmt == 'Percentual': y_params['tickformat'] = '.0%'
            elif fmt == 'Moeda': 
                y_params['tickformat'] = ',.2f'
                y_params['tickprefix'] = 'R$ '
            elif fmt == 'Milhar': y_params['tickformat'] = '.2s'
            elif fmt == 'Milhões': y_params['tickformat'] = '.2s' 
            if y_params: fig.update_yaxes(**y_params)

        return fig

    @staticmethod
    def create_altair_plot(df, x, y):
        if alt is not None: return alt.Chart(df.to_pandas()).mark_circle(size=60).encode(x=x, y=y, tooltip=[x, y]).interactive()
        return None

    @staticmethod
    def get_echarts_options(x_data, y_data):
        return {"xAxis": {"type": "category", "data": x_data}, "yAxis": {"type": "value"}, "series": [{"data": y_data, "type": "line", "smooth": True}]}

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

class MLProcessor:
    @staticmethod
    def run_kmeans(df, features, n_clusters):
        data = df.select(features).to_pandas().select_dtypes(include=[np.number]).dropna()
        if data.empty: return None
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(data)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        return kmeans.fit_predict(scaled_data)

class TemplateLibrary:
    TEMPLATES = ["Em Branco", "Dashboard Financeiro", "Dashboard Comercial", "Dashboard RH", "Dashboard Logístico", "Dashboard Educacional", "Dashboard Ambiental"]

class UITheme:
    @staticmethod
    def apply_custom_css():
        st.markdown("""
        <style>
            .main { background-color: #f8f9fa; }
            .kpi-card { background: white; padding: 15px; border-radius: 10px; border-left: 5px solid #007bff; box-shadow: 0 2px 4px rgba(0,0,0,0.05); text-align: center; height: 100%; display: flex; flex-direction: column; justify-content: center; }
            .kpi-value { font-size: 24px; font-weight: bold; color: #2c3e50; }
            .kpi-label { font-size: 12px; color: #7f8c8d; text-transform: uppercase; margin-bottom: 5px;}
            div[role="radiogroup"] { padding-bottom: 20px; border-bottom: 2px solid #e0e0e0; }
        </style>
        """, unsafe_allow_html=True)

    @staticmethod
    def render_kpi_html(label, value):
        return f"""<div class="kpi-card"><div class="kpi-label">{label}</div><div class="kpi-value">{value}</div></div>"""

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
        self.templates = TemplateLibrary()
        self.repo = ProjectRepository()
        HistoryManager.init_state()

    def run(self):
        self.theme.apply_custom_css()
        with st.sidebar:
            st.title("📂 DataViz Pro V2.0")
            uploaded_file = st.file_uploader("Carregar dataset", type=["csv", "xlsx", "json", "parquet"])
            st.divider()
            st.subheader("Gerenciador de Projetos")
            if st.button("Carregar Lista de Projetos"):
                st.session_state['proj_df'] = self.repo.load_projects()
            
            if 'proj_df' in st.session_state and not st.session_state['proj_df'].empty:
                st.dataframe(st.session_state['proj_df'], use_container_width=True)
                pid_load = st.text_input("ID para Abrir:")
                if st.button("Abrir Projeto"):
                    loaded_state = self.repo.load_canvas_state(pid_load)
                    if loaded_state:
                        HistoryManager.push_state(loaded_state)
                        st.success("Projeto carregado!")
        
        st.title("📊 Painel de Análise Profissional")
        if uploaded_file:
            df = self.engine.load_data(uploaded_file, uploaded_file.name)
            if df is not None:
                schema = self.engine.get_semantic_schema(df)
                
                abas_lista = ["Dashboard", "Exploratória", "Estúdio Visual", "Transformação", "Clusterização", "Anomalias", "Assistente IA", "Smart Profiling", "PyGWalker", "Canvas Visual"]
                
                if 'aba_atual' not in st.session_state: st.session_state['aba_atual'] = "Dashboard"

                def mudar_aba():
                    st.session_state['aba_atual'] = st.session_state['nav_radio']

                st.radio("Navegação do Painel:", abas_lista, horizontal=True, key="nav_radio", index=abas_lista.index(st.session_state['aba_atual']), on_change=mudar_aba, label_visibility="collapsed")
                aba_ativa = st.session_state['aba_atual']
                
                if aba_ativa == "Dashboard":
                    st.subheader("Dashboard Executivo")
                    st.metric("Qualidade dos Dados", f"{DataProfiler.calculate_quality_score(df)}/100")
                    st.info(self.insight.generate_summary(df))
                    cols = [c for c, t in schema.items() if t == "Métrica"]
                    if cols:
                        cols_ui = st.columns(min(len(cols), 4))
                        for i, m in enumerate(cols[:4]):
                            with cols_ui[i]: st.markdown(self.theme.render_kpi_html(m, f"{df[m].sum():,.0f}"), unsafe_allow_html=True)
                    st.plotly_chart(self.viz.auto_plot(df, schema), use_container_width=True)

                elif aba_ativa == "Exploratória":
                    st.subheader("Configuração Manual de Gráficos")
                    c1, c2 = st.columns(2)
                    x = c1.selectbox("Eixo X (Manual)", list(df.columns), key="expl_x")
                    y = c2.selectbox("Eixo Y (Manual)", list(df.columns), key="expl_y")
                    st.plotly_chart(self.viz.create_plot(df, "Barras", x, y), use_container_width=True)

                elif aba_ativa == "Estúdio Visual":
                    st.subheader("Estúdio de Visualização Avançado")
                    c1, c2, c3 = st.columns([1, 3, 1])
                    with c1:
                        gx = st.selectbox("Eixo X", list(df.columns), key="studio_x")
                        gy = st.selectbox("Eixo Y", list(df.columns), key="studio_y")
                        gtype = st.selectbox("Tipo", ["Barras", "Linha", "Dispersão", "Histograma", "Treemap", "Sunburst", "Area", "Radar", "Polar", "Funnel", "Waterfall", "Violin", "Boxplot", "Heatmap", "Density Map", "Altair", "ECharts"], key="studio_type")
                        
                        st.divider()
                        st.write("🔧 Editor de Propriedades Reais")
                        layout_title = st.text_input("Título Gráfico", "Meu Gráfico")
                        bg_color = st.color_picker("Cor Fundo (Plot)", "#FFFFFF")
                        
                        st.write("📏 Editor de Eixos")
                        format_y = st.selectbox("Formato (Eixo Y)", ["Linear", "Percentual", "Moeda", "Milhar", "Milhões"])
                        scale_y = st.selectbox("Escala (Eixo Y)", ["Linear", "Log"])
                        inv_x = st.checkbox("Inverter Eixo X")
                        inv_y = st.checkbox("Inverter Eixo Y")

                    with c2: 
                        config = {
                            "title": layout_title, "plot_bgcolor": bg_color, "paper_bgcolor": "#f8f9fa",
                            "format_y": format_y, "scale_y": scale_y, "invert_x": inv_x, "invert_y": inv_y
                        }
                        
                        if gtype == "Altair" and alt is not None: st.altair_chart(self.viz.create_altair_plot(df, gx, gy), use_container_width=True)
                        elif gtype == "ECharts" and st_echarts is not None: st_echarts(options=self.viz.get_echarts_options(df.to_pandas()[gx].tolist()[:50], df.to_pandas()[gy].tolist()[:50]), height="400px")
                        else: st.plotly_chart(self.viz.create_plot(df, gtype, gx, gy, config=config), use_container_width=True)

                elif aba_ativa == "Transformação":
                    st.subheader("Engenharia de Dados")
                    if st.button("Remover Duplicados", key="dup_btn"): df = self.transformer.remove_duplicates(df)
                    if st.button("Preencher Nulos", key="null_btn"): df = self.transformer.fill_nulls(df)
                    st.dataframe(df.to_pandas().head())

                elif aba_ativa == "Clusterização":
                    num_cols = [c for c, t in schema.items() if t == "Métrica"]
                    sel = st.multiselect("Features", num_cols, default=num_cols[:2], key="ml_sel")
                    k = st.slider("Clusters", 2, 6, 3, key="ml_k")
                    if st.button("Executar Clusterização", key="ml_btn"):
                        res = self.ml.run_kmeans(df, sel, k)
                        if res is not None: st.write("Clusterização finalizada.")

                elif aba_ativa == "Anomalias":
                    st.subheader("Detecção de Anomalias")
                    num_cols = [c for c, t in schema.items() if t == "Métrica"]
                    sel = st.multiselect("Features Anomalia", num_cols, default=num_cols[:2], key="ano_sel")
                    if st.button("Rodar Detecção", key="ano_btn"):
                        preds = AnomalyDetector.find_anomalies(df, sel)
                        pdf = df.to_pandas()
                        pdf['Status'] = ["Anômalo" if p == -1 else "Normal" for p in preds]
                        st.dataframe(pdf[pdf['Status'] == "Anômalo"])

                elif aba_ativa == "Assistente IA":
                    query = st.text_input("Consulta SQL:", key="sql_input")
                    if query: st.dataframe(self.engine.query(query))
                    if prompt := st.chat_input("Pergunte algo...", key="ai_chat"):
                        st.chat_message("user").markdown(prompt)
                        st.chat_message("assistant").markdown("Motor de IA ativo.")

                elif aba_ativa == "Smart Profiling":
                    st.subheader("Motor Inteligente: Profiling Automático")
                    if ydata_profiling is not None:
                        with st.spinner("Analisando..."):
                            components.html(df.to_pandas().profile_report().to_html(), height=800, scrolling=True)
                    else: st.error("Instale 'ydata-profiling'.")
                            
                elif aba_ativa == "PyGWalker":
                    st.subheader("Tableau-like Experience (PyGWalker)")
                    if pyg is not None: components.html(pyg.to_html(df.to_pandas()), height=1000, scrolling=True)
                    else: st.error("Instale 'pygwalker'.")

                elif aba_ativa == "Canvas Visual":
                    st.subheader("🎨 Estúdio Canvas Pro")
                    
                    top_c1, top_c2, top_c3, top_c4 = st.columns([1, 1, 2, 8])
                    
                    with top_c1: 
                        if st.button("↩ Undo"): HistoryManager.undo()
                    with top_c2:
                        if st.button("↪ Redo"): HistoryManager.redo()
                    with top_c3:
                        if st.button("💾 Salvar Projeto"):
                            pid = self.repo.save_canvas_state("Meu_Projeto_Canvas", st.session_state['canvas_objects'])
                            st.success(f"Salvo! ID: {pid[:8]}")

                    col_tools, col_canvas = st.columns([2, 8])
                    
                    with col_tools:
                        st.write("**Ferramentas de Criação**")
                        if st.button("➕ Adicionar Gráfico", use_container_width=True):
                            new_obj = {"id": str(uuid.uuid4()), "type": "chart", "x": 0, "y": 0, "w": 6, "h": 4, "config": {"type": "Barras", "x": df.columns[0], "y": df.columns[1] if len(df.columns)>1 else df.columns[0]}}
                            current_objs = copy.deepcopy(st.session_state['canvas_objects'])
                            current_objs.append(new_obj)
                            HistoryManager.push_state(current_objs)
                            
                        if st.button("➕ Adicionar KPI", use_container_width=True):
                            new_obj = {"id": str(uuid.uuid4()), "type": "kpi", "x": 0, "y": 0, "w": 3, "h": 2, "config": {"label": "Novo KPI", "val": "0"}}
                            current_objs = copy.deepcopy(st.session_state['canvas_objects'])
                            current_objs.append(new_obj)
                            HistoryManager.push_state(current_objs)

                        st.divider()
                        
                        # CORREÇÃO CIRÚRGICA 2: Isolamento da Fila de Estado (pending_update)
                        st.write("**Elementos Ativos (Editor)**")
                        for idx, obj in enumerate(st.session_state['canvas_objects']):
                            with st.expander(f"{obj['type'].upper()} ({obj['id'][:5]})"):
                                if obj['type'] == 'kpi':
                                    new_lbl = st.text_input("Label", obj['config']['label'], key=f"lbl_{obj['id']}")
                                    new_val = st.text_input("Valor", obj['config']['val'], key=f"val_{obj['id']}")
                                    if new_lbl != obj['config']['label'] or new_val != obj['config']['val']:
                                        mod_objs = copy.deepcopy(st.session_state['canvas_objects'])
                                        mod_objs[idx]['config']['label'] = new_lbl
                                        mod_objs[idx]['config']['val'] = new_val
                                        st.session_state["pending_update"] = mod_objs
                                elif obj['type'] == 'chart':
                                    st.write("Propriedades ligadas ao painel de eixos...")
                        
                        if "pending_update" in st.session_state:
                            HistoryManager.push_state(st.session_state["pending_update"])
                            del st.session_state["pending_update"]

                    with col_canvas:
                        if elements is not None:
                            canvas_items = st.session_state['canvas_objects']
                            if not canvas_items:
                                st.info("O Canvas está vazio. Adicione elementos no painel lateral.")
                            else:
                                try:
                                    # CORREÇÃO CIRÚRGICA 3: Forçamento de Remount via Hash Único Dinâmico
                                    with elements(f"canvas_{hash(str(canvas_items))}"):
                                        layout = [dashboard.Item(obj["id"], obj["x"], obj["y"], obj["w"], obj["h"]) for obj in canvas_items]
                                        with dashboard.Grid(layout):
                                            for obj in canvas_items:
                                                # CORREÇÃO CIRÚRGICA 1: Chaves Exclusivas (dashboard.Item vs mui.Paper)
                                                with mui.Paper(key=f"paper_{obj['id']}", elevation=3, sx={"p": 2, "display": "flex", "flexDirection": "column", "justifyContent": "center"}):
                                                    if obj["type"] == "kpi":
                                                        # CORREÇÃO CIRÚRGICA 4: Simplificação MUI para evitar bugs de aninhamento
                                                        mui.Typography(f"{obj['config']['label']} - {obj['config']['val']}", variant="h6", sx={"fontWeight": "bold", "color": "#2c3e50", "textAlign": "center", "mt": 2})
                                                    elif obj["type"] == "chart":
                                                        mui.Typography("Gráfico Dinâmico Plotly", variant="caption", sx={"textAlign": "center"})
                                except Exception as e:
                                    st.exception(e)
                        else:
                            st.error("O pacote 'streamlit-elements' não está instalado.")
        else:
            st.info("Carregue um arquivo para iniciar.")

if __name__ == "__main__":
    app = DataVizApp()
    app.run()
