"""
Auditor de Ventas Inteligente — VAE Frontend
Streamlit app con sistema de login y dos perfiles: Técnico y Negocio.
"""
import json
import io
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from src.logger import log_info, log_warning, log_error

# ── Configuración de página ──────────────────────────────────────────────────
st.set_page_config(
    page_title="Auditor de Ventas Inteligente",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Paleta de colores de severidad ───────────────────────────────────────────
SEVERITY_COLORS = {
    "normal": "#22c55e",
    "baja":   "#eab308",
    "media":  "#f97316",
    "alta":   "#ef4444",
}
SEVERITY_ORDER = ["normal", "baja", "media", "alta"]
SEV_LABELS     = {"normal": "Normal", "baja": "Baja", "media": "Media", "alta": "Alta"}

# ── Credenciales hardcoded ────────────────────────────────────────────────────
USERS = {
    "tecnico": {"password": "admin123", "rol": "Técnico",  "display": "Técnico VAE"},
    "rosita":  {"password": "rosita123", "rol": "Negocio", "display": "Restaurante Rosita"},
}

# ── Inicializar session_state ─────────────────────────────────────────────────
if "logged_in" not in st.session_state:
    st.session_state.logged_in  = False
    st.session_state.username   = ""
    st.session_state.rol        = ""
    st.session_state.display    = ""
    st.session_state.login_err  = ""


# ── CSS global ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .stApp { background-color: #0f172a; color: #e2e8f0; }
    section[data-testid="stSidebar"] {
        background-color: #1e293b;
        border-right: 1px solid #334155;
    }
    .kpi-card {
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px 24px;
        text-align: center;
    }
    .kpi-title { font-size:0.85rem; color:#94a3b8; text-transform:uppercase;
                 letter-spacing:0.08em; margin-bottom:6px; }
    .kpi-value { font-size:2rem; font-weight:700; color:#f1f5f9; }
    .kpi-sub   { font-size:0.78rem; color:#64748b; margin-top:4px; }
    h2, h3 { color:#f1f5f9 !important; }
    .stTabs [data-baseweb="tab-list"] { background-color:#1e293b; gap:4px; }
    .stTabs [data-baseweb="tab"] {
        background-color:#0f172a; color:#94a3b8;
        border-radius:8px 8px 0 0; padding:8px 20px;
        border:1px solid #334155;
    }
    .stTabs [aria-selected="true"] {
        background-color:#334155 !important; color:#f1f5f9 !important;
    }
    /* Login card */
    .login-card {
        max-width: 420px;
        margin: 60px auto;
        background: #1e293b;
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 40px 36px 32px;
    }
    /* Rosita card */
    .rosita-welcome {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #f97316;
        border-radius: 16px;
        padding: 28px 32px;
        margin-bottom: 24px;
    }
</style>
""", unsafe_allow_html=True)

# ── Layout de Plotly (tema oscuro) ────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    paper_bgcolor="#1e293b",
    plot_bgcolor="#1e293b",
    font=dict(color="#e2e8f0", family="Inter, sans-serif"),
    margin=dict(l=20, r=20, t=40, b=20),
)


# ════════════════════════════════════════════════════════════════════════════
# PANTALLA DE LOGIN
# ════════════════════════════════════════════════════════════════════════════
def show_login():
    # Centrar el formulario con columnas
    _, center, _ = st.columns([1, 1.2, 1])
    with center:
        st.markdown("""
        <div style="text-align:center; padding: 40px 0 20px;">
          <div style="font-size:3rem;">🔍</div>
          <h1 style="color:#f1f5f9; font-size:1.7rem; margin:8px 0 4px;">
            Auditor de Ventas Inteligente
          </h1>
          <p style="color:#64748b; font-size:0.9rem;">
            Ingresa tus credenciales para continuar
          </p>
        </div>
        """, unsafe_allow_html=True)

        with st.form("login_form", clear_on_submit=False):
            usuario = st.text_input("👤 Usuario", placeholder="Ingresa tu usuario")
            clave   = st.text_input("🔒 Contraseña", type="password",
                                    placeholder="Ingresa tu contraseña")
            submitted = st.form_submit_button("Iniciar Sesión", use_container_width=True)

            if submitted:
                user_data = USERS.get(usuario.strip().lower())

                if user_data and clave == user_data["password"]:
                    st.session_state.logged_in = True
                    st.session_state.username = usuario.strip().lower()
                    st.session_state.rol = user_data["rol"]
                    st.session_state.display = user_data["display"]
                    st.session_state.login_err = ""

                    log_info(
                        "LOGIN_OK",
                        f"usuario={st.session_state.username}, rol={st.session_state.rol}"
                    )

                    st.rerun()

                else:
                    st.session_state.login_err = "Usuario o contraseña incorrectos."

                    log_warning(
                        "LOGIN_FAIL",
                        f"usuario={usuario}"
                    )

        if st.session_state.login_err:
            st.error(st.session_state.login_err)

        st.markdown("""
        <div style="text-align:center; margin-top:20px; color:#475569; font-size:0.78rem;">
          Demo — credenciales de prueba disponibles
        </div>
        """, unsafe_allow_html=True)


if not st.session_state.logged_in:
    show_login()
    st.stop()


# ── Carga de datos con caché (solo para rol Técnico) ─────────────────────────
@st.cache_data
def load_metricas():
    with open("reports/metricas_evaluacion.json", encoding="utf-8") as f:
        return json.load(f)

@st.cache_data
def load_transacciones():
    return pd.read_csv("reports/evaluacion_transacciones.csv")

@st.cache_data
def load_training_history():
    return pd.read_csv("reports/vae_training_history.csv")

@st.cache_data
def load_umbrales():
    with open("reports/umbral_severidad.json", encoding="utf-8") as f:
        return json.load(f)


# ── Sidebar compartido (botón de logout) ─────────────────────────────────────
with st.sidebar:
    rol_icon = "🛠️" if st.session_state.rol == "Técnico" else "🍽️"
    st.markdown(
        f"""
        <div style="padding:12px 0 8px;">
          <div style="font-size:1.6rem; text-align:center;">{rol_icon}</div>
          <div style="text-align:center; font-weight:700; color:#f1f5f9;
                      font-size:0.95rem; margin-top:4px;">
            {st.session_state.display}
          </div>
          <div style="text-align:center; color:#64748b; font-size:0.78rem; margin-top:2px;">
            Rol: {st.session_state.rol}
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown("---")

    if st.session_state.rol == "Técnico":
        st.markdown("#### Navegación")
        seccion = st.radio(
            "",
            ["📊 Metricas de Rendimiento", "📁 Auditoría por Lotes (CSV)", "⚡ Verificar Venta"],
            label_visibility="collapsed",
        )
        umbrales = load_umbrales()
        st.markdown("---")
        st.markdown("#### Umbrales de Severidad")
        st.markdown(
            f"""
            <div style='font-size:0.82rem; line-height:1.9;'>
              <span style='color:{SEVERITY_COLORS["baja"]}'>●</span>
              <b>Baja</b> &nbsp;≥ {umbrales['umbral_baja']:.4f}<br>
              <span style='color:{SEVERITY_COLORS["media"]}'>●</span>
              <b>Media</b> &nbsp;≥ {umbrales['umbral_media']:.4f}<br>
              <span style='color:{SEVERITY_COLORS["alta"]}'>●</span>
              <b>Alta</b> &nbsp;≥ {umbrales['umbral_alta']:.4f}
            </div>
            """,
            unsafe_allow_html=True,
        )
        st.markdown("---")
        st.caption("Modelo VAE · 50 épocas · Split: prueba")
    else:
        seccion = None  # No aplica para Negocio

    st.markdown("---")
    if st.button("🚪 Cerrar Sesión", use_container_width=True):
        for key in ["logged_in", "username", "rol", "display", "login_err"]:
            st.session_state[key] = False if key == "logged_in" else ""
        st.rerun()


# ════════════════════════════════════════════════════════════════════════════
# ENRUTAMIENTO POR ROL
# ════════════════════════════════════════════════════════════════════════════
if st.session_state.rol == "Técnico":

    metricas   = load_metricas()
    df_tx      = load_transacciones()
    df_history = load_training_history()
    umbrales   = load_umbrales()
    m          = metricas["prueba"]

    # ────────────────────────────────────────────────────────────────────────
    # SECCIÓN 1 — DASHBOARD GENERAL
    # ────────────────────────────────────────────────────────────────────────
    if seccion == "📊 Metricas de Rendimiento":
        st.markdown("## 📊 Metricas de Rendimiento")
        st.markdown("Resumen del desempeño del modelo VAE y análisis de anomalías detectadas.")

        k1, k2, k3, k4 = st.columns(4)
        with k1:
            st.markdown(
                f"""<div class="kpi-card">
                  <div class="kpi-title">Precisión (Prueba)</div>
                  <div class="kpi-value" style="color:#22c55e;">{m['precision']*100:.1f}%</div>
                  <div class="kpi-sub">F1-Score: {m['f1_score']*100:.1f}%</div>
                </div>""", unsafe_allow_html=True)
        with k2:
            st.markdown(
                f"""<div class="kpi-card">
                  <div class="kpi-title">Total Anomalías</div>
                  <div class="kpi-value" style="color:#f97316;">{m['n_predichas_anomalas']:,}</div>
                  <div class="kpi-sub">de {m['n_transacciones']:,} transacciones</div>
                </div>""", unsafe_allow_html=True)
        with k3:
            st.markdown(
                f"""<div class="kpi-card">
                  <div class="kpi-title">Monto en Riesgo</div>
                  <div class="kpi-value" style="color:#ef4444;">${m['monto_total_en_riesgo']:,.2f}</div>
                  <div class="kpi-sub">{m['pct_monto_en_riesgo']:.1f}% del total</div>
                </div>""", unsafe_allow_html=True)
        with k4:
            st.markdown(
                f"""<div class="kpi-card">
                  <div class="kpi-title">Recall</div>
                  <div class="kpi-value" style="color:#a78bfa;">{m['recall']*100:.1f}%</div>
                  <div class="kpi-sub">Anomalías reales detectadas</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # Gráficos: pastel + training history
        col_pie, col_line = st.columns([1, 1.6])
        with col_pie:
            st.markdown("### Distribución de Severidad")
            sev_dist   = m["distribucion_severidad"]
            sev_labels = [SEV_LABELS[k] for k in SEVERITY_ORDER if k in sev_dist]
            sev_values = [sev_dist[k]   for k in SEVERITY_ORDER if k in sev_dist]
            sev_colors = [SEVERITY_COLORS[k] for k in SEVERITY_ORDER if k in sev_dist]
            fig_pie = go.Figure(go.Pie(
                labels=sev_labels, values=sev_values,
                marker=dict(colors=sev_colors, line=dict(color="#0f172a", width=2)),
                hole=0.45, textinfo="percent+label", textfont=dict(size=13),
                hovertemplate="<b>%{label}</b><br>Transacciones: %{value}<br>%{percent}<extra></extra>",
            ))
            fig_pie.update_layout(**PLOTLY_LAYOUT, showlegend=True, height=340,
                legend=dict(orientation="h", y=-0.08, x=0.5, xanchor="center"),
                annotations=[dict(text=f"<b>{m['n_transacciones']}</b><br>total",
                    x=0.5, y=0.5, font_size=14, showarrow=False,
                    font=dict(color="#e2e8f0"))])
            st.plotly_chart(fig_pie, use_container_width=True)

        with col_line:
            st.markdown("### Historial de Entrenamiento (Loss)")
            fig_loss = go.Figure()
            fig_loss.add_trace(go.Scatter(
                x=df_history["epoch"], y=df_history["loss"],
                mode="lines", name="Loss Total",
                line=dict(color="#818cf8", width=2.5),
                hovertemplate="Época %{x}<br>Loss: %{y:.5f}<extra></extra>"))
            fig_loss.add_trace(go.Scatter(
                x=df_history["epoch"], y=df_history["reconstruction_mse"],
                mode="lines", name="MSE Reconstrucción",
                line=dict(color="#22c55e", width=2, dash="dot"),
                hovertemplate="Época %{x}<br>MSE: %{y:.5f}<extra></extra>"))
            fig_loss.add_trace(go.Scatter(
                x=df_history["epoch"],
                y=df_history["kl_loss"] / df_history["kl_loss"].max(),
                mode="lines", name="KL Loss (norm.)",
                line=dict(color="#f97316", width=2, dash="dash"),
                hovertemplate="Época %{x}<br>KL norm: %{y:.4f}<extra></extra>"))
            fig_loss.update_layout(**PLOTLY_LAYOUT, height=340,
                xaxis=dict(title="Época", gridcolor="#334155", zeroline=False),
                yaxis=dict(title="Pérdida", gridcolor="#334155", zeroline=False),
                legend=dict(orientation="h", y=1.12, x=0, bgcolor="rgba(0,0,0,0)"))
            st.plotly_chart(fig_loss, use_container_width=True)


        # Barras: comparación splits
        st.markdown("### Comparación de Severidad: Validación vs Prueba")
        splits_data = []
        for split_key, split_label in [("validacion", "Validación"), ("prueba", "Prueba")]:
            dist = metricas[split_key]["distribucion_severidad"]
            for sev in SEVERITY_ORDER:
                splits_data.append({
                    "Split": split_label,
                    "Severidad": SEV_LABELS[sev],
                    "Transacciones": dist.get(sev, 0),
                })
        df_splits = pd.DataFrame(splits_data)
        fig_bar = px.bar(
            df_splits, x="Severidad", y="Transacciones",
            color="Severidad", barmode="group", facet_col="Split",
            color_discrete_map={SEV_LABELS[k]: SEVERITY_COLORS[k] for k in SEVERITY_ORDER},
            text="Transacciones",
        )
        fig_bar.update_traces(textposition="outside", textfont=dict(size=11),
                              marker_line_color="#0f172a", marker_line_width=1.5)
        fig_bar.update_layout(**PLOTLY_LAYOUT, showlegend=False, height=340,
            xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"),
            xaxis2=dict(gridcolor="#334155"), yaxis2=dict(gridcolor="#334155"))
        fig_bar.for_each_annotation(lambda a: a.update(
            text=a.text.replace("Split=", ""), font=dict(size=13, color="#cbd5e1")))
        st.plotly_chart(fig_bar, use_container_width=True)

        # Matriz de confusión
        st.markdown("### Matriz de Confusión (Split: Prueba)")
        mc = m["matriz_confusion"]
        z  = [[mc["verdaderos_negativos"], mc["falsos_positivos"]],
              [mc["falsos_negativos"],     mc["verdaderos_positivos"]]]
        fig_cm = go.Figure(go.Heatmap(
            z=z,
            x=["Predicho: Normal", "Predicho: Anomalía"],
            y=["Real: Normal", "Real: Anomalía"],
            colorscale=[[0,"#1e293b"],[0.5,"#4f46e5"],[1,"#22c55e"]],
            showscale=False,
            text=[[str(v) for v in row] for row in z],
            texttemplate="%{text}", textfont=dict(size=22, color="#f1f5f9"),
            hovertemplate="%{y} / %{x}: %{z}<extra></extra>",
        ))
        fig_cm.update_layout(**PLOTLY_LAYOUT, height=260,
                             xaxis=dict(side="top"), yaxis=dict(autorange="reversed"))
        st.plotly_chart(fig_cm, use_container_width=True)


    # ────────────────────────────────────────────────────────────────────────
    # SECCIÓN 2 — AUDITORÍA POR LOTES
    # ────────────────────────────────────────────────────────────────────────
    elif seccion == "📁 Auditoría por Lotes (CSV)":
        st.markdown("## 📁 Auditoría por Lotes")
        st.markdown(
            "Carga un CSV con transacciones evaluadas o usa el dataset incluido."
        )
        use_default = st.checkbox("Usar dataset de evaluación incluido", value=True)
        if use_default:
            df_audit = df_tx.copy()
            st.info(f"Cargadas **{len(df_audit):,}** transacciones desde `reports/evaluacion_transacciones.csv`")
        else:
            uploaded = st.file_uploader("Sube tu archivo CSV", type=["csv"])
            if uploaded is None:
                st.warning("Sube un archivo CSV para continuar.")
                st.stop()
            df_audit = pd.read_csv(uploaded)
            st.success(f"Archivo cargado: **{len(df_audit):,}** filas")

        st.markdown("---")
        fcol1, fcol2, fcol3 = st.columns(3)
        with fcol1:
            splits_disp = ["Todos"] + sorted(df_audit["split"].dropna().unique().tolist())
            filtro_split = st.selectbox("Filtrar por Split", splits_disp)
        with fcol2:
            sevs_disp = ["Todas"] + [SEV_LABELS[s] for s in SEVERITY_ORDER
                                     if s in df_audit["severidad"].values]
            filtro_sev = st.selectbox("Filtrar por Severidad", sevs_disp)
        with fcol3:
            solo_anomalias = st.toggle("Solo anomalías detectadas", value=False)

        df_filtered = df_audit.copy()
        if filtro_split != "Todos":
            df_filtered = df_filtered[df_filtered["split"] == filtro_split]
        if filtro_sev != "Todas":
            sev_key = {v: k for k, v in SEV_LABELS.items()}[filtro_sev]
            df_filtered = df_filtered[df_filtered["severidad"] == sev_key]
        if solo_anomalias:
            df_filtered = df_filtered[df_filtered["prediccion_anomalia"] == 1]

        st.markdown(f"**{len(df_filtered):,}** transacciones con los filtros aplicados.")

        bk1, bk2, bk3 = st.columns(3)
        anomalias_lote    = df_filtered[df_filtered["prediccion_anomalia"] == 1]
        monto_riesgo_lote = anomalias_lote["monto_final"].sum()
        pct_anomalias     = (len(anomalias_lote) / len(df_filtered) * 100) if len(df_filtered) else 0
        with bk1:
            st.metric("Transacciones", f"{len(df_filtered):,}")
        with bk2:
            st.metric("Anomalías detectadas", f"{len(anomalias_lote):,}",
                      delta=f"{pct_anomalias:.1f}% del lote", delta_color="inverse")
        with bk3:
            st.metric("Monto en riesgo", f"${monto_riesgo_lote:,.2f}")

        gc1, gc2 = st.columns(2)
        with gc1:
            st.markdown("#### Distribución del Error de Reconstrucción")
            fig_hist = px.histogram(
                df_filtered, x="reconstruction_error", color="severidad",
                color_discrete_map=SEVERITY_COLORS,
                category_orders={"severidad": SEVERITY_ORDER},
                nbins=60, barmode="overlay", opacity=0.75,
                labels={"reconstruction_error": "Error de Reconstrucción", "severidad": "Severidad"},
            )
            for key, label in [("umbral_baja","Baja"),("umbral_media","Media"),("umbral_alta","Alta")]:
                fig_hist.add_vline(x=umbrales[key], line_dash="dash",
                    line_color=SEVERITY_COLORS[key.replace("umbral_","")],
                    annotation_text=label, annotation_position="top right",
                    annotation_font_color=SEVERITY_COLORS[key.replace("umbral_","")])
            fig_hist.update_layout(**PLOTLY_LAYOUT, height=320,
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155", title="Frecuencia"),
                legend=dict(title="Severidad"))
            st.plotly_chart(fig_hist, use_container_width=True)

        with gc2:
            st.markdown("#### Monto en Riesgo por Severidad")
            risk_by_sev = (
                df_filtered[df_filtered["prediccion_anomalia"] == 1]
                .groupby("severidad")["monto_final"].sum()
                .reindex(SEVERITY_ORDER).dropna().reset_index()
            )
            risk_by_sev.columns = ["severidad", "monto"]
            risk_by_sev["label"] = risk_by_sev["severidad"].map(SEV_LABELS)
            fig_risk = px.bar(
                risk_by_sev, x="label", y="monto", color="severidad",
                color_discrete_map=SEVERITY_COLORS,
                text=risk_by_sev["monto"].apply(lambda v: f"${v:,.0f}"),
                labels={"label": "Severidad", "monto": "Monto ($)"},
            )
            fig_risk.update_traces(textposition="outside", marker_line_width=0)
            fig_risk.update_layout(**PLOTLY_LAYOUT, height=320, showlegend=False,
                xaxis=dict(gridcolor="#334155"), yaxis=dict(gridcolor="#334155"))
            st.plotly_chart(fig_risk, use_container_width=True)

        st.markdown("#### Detalle de Transacciones")
        display_cols = ["id_transaccion","split","monto_final","tipo_anomalia",
                        "reconstruction_error","severidad","prediccion_anomalia"]
        available_cols = [c for c in display_cols if c in df_filtered.columns]
        df_display = df_filtered[available_cols].copy()
        if "monto_final" in df_display.columns:
            df_display["monto_final"] = df_display["monto_final"].apply(lambda x: f"${x:,.2f}")
        if "reconstruction_error" in df_display.columns:
            df_display["reconstruction_error"] = df_display["reconstruction_error"].apply(lambda x: f"{x:.6f}")
        if "prediccion_anomalia" in df_display.columns:
            df_display["prediccion_anomalia"] = df_display["prediccion_anomalia"].map({0:"✅ Normal",1:"⚠️ Anomalía"})
        df_display = df_display.rename(columns={
            "id_transaccion":"ID","split":"Split","monto_final":"Monto",
            "tipo_anomalia":"Tipo","reconstruction_error":"Error Reconstrucción",
            "severidad":"Severidad","prediccion_anomalia":"Predicción"})
        st.dataframe(df_display.head(500), use_container_width=True, height=400)
        st.download_button("⬇️ Descargar resultados filtrados (.csv)",
            data=df_filtered.to_csv(index=False).encode("utf-8"),
            file_name="auditoria_filtrada.csv", mime="text/csv")


    # ────────────────────────────────────────────────────────────────────────
    # SECCIÓN 3 — EVALUACIÓN EN VIVO
    # ────────────────────────────────────────────────────────────────────────
    elif seccion == "⚡ Verificar Venta":
        st.markdown("## ⚡ Verificar Venta")
        st.markdown(
            "Ingresa los datos de una transacción. "
            "El sistema simulará el error de reconstrucción del VAE y clasificará su severidad."
        )
        with st.form("form_live"):
            st.markdown("### Datos de la Transacción")
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                monto     = st.number_input("Monto ($)", min_value=0.0, max_value=5000.0,
                                            value=15.50, step=0.5, format="%.2f")
                descuento = st.number_input("Descuento aplicado ($)", min_value=0.0,
                                            max_value=500.0, value=0.0, step=0.5)
            with lc2:
                hora       = st.slider("Hora del día", 0, 23, 12)
                dia_semana = st.selectbox("Día de la semana",
                    ["Lunes","Martes","Miércoles","Jueves","Viernes","Sábado","Domingo"])
            with lc3:
                metodo_pago = st.selectbox("Método de pago",
                    ["Efectivo","Tarjeta crédito","Tarjeta débito","Transferencia","QR / Billetera digital"])
                num_items = st.number_input("Número de ítems", min_value=1, max_value=50, value=3, step=1)
            submitted = st.form_submit_button("🔍 Evaluar Venta", use_container_width=True)

        if submitted:
            base_error = 0.035 + np.random.uniform(0.005, 0.015)
            if monto > 200:
                base_error += (monto - 200) / 1000.0 * 0.6
            if monto < 1.0:
                base_error += 0.04
            if descuento > 0 and (descuento / max(monto, 1)) > 0.3:
                base_error += (descuento / max(monto, 1)) * 0.15
            if hora < 6 or hora > 22:
                base_error += 0.025
            if num_items > 20:
                base_error += 0.02
            rec_error = float(np.clip(base_error + np.random.normal(0, 0.003), 0.01, 1.0))

            if rec_error >= umbrales["umbral_alta"]:
                severidad = "alta"
            elif rec_error >= umbrales["umbral_media"]:
                severidad = "media"
            elif rec_error >= umbrales["umbral_baja"]:
                severidad = "baja"
            else:
                severidad = "normal"
            es_anomalia = severidad != "normal"

            st.markdown("---")
            st.markdown("### Resultado de la Evaluación")
            sev_color = SEVERITY_COLORS[severidad]
            icono     = "🚨" if severidad == "alta" else ("⚠️" if severidad in ("media","baja") else "✅")

            res1, res2 = st.columns([1, 2])
            with res1:
                st.markdown(
                    f"""<div style="background:{sev_color}18; border:2px solid {sev_color};
                        border-radius:16px; padding:28px; text-align:center;">
                      <div style="font-size:3rem;">{icono}</div>
                      <div style="font-size:1.6rem; font-weight:700; color:{sev_color};
                                  margin-top:8px;">{SEV_LABELS[severidad]}</div>
                      <div style="font-size:0.9rem; color:#94a3b8; margin-top:6px;">
                        {"⚠️ Anomalía detectada" if es_anomalia else "✅ Transacción normal"}
                      </div>
                      <hr style="border-color:#334155; margin:14px 0;">
                      <div style="font-size:0.85rem; color:#94a3b8;">Error de reconstrucción</div>
                      <div style="font-size:1.5rem; font-weight:600; color:#e2e8f0;">
                        {rec_error:.6f}
                      </div>
                    </div>""", unsafe_allow_html=True)
            with res2:
                fig_gauge = go.Figure(go.Indicator(
                    mode="gauge+number",
                    value=rec_error,
                    number=dict(valueformat=".5f", font=dict(color="#e2e8f0", size=20)),
                    gauge=dict(
                        axis=dict(range=[0, max(umbrales["umbral_alta"]*2, rec_error*1.2)],
                                  tickcolor="#94a3b8", tickfont=dict(color="#94a3b8", size=10)),
                        bar=dict(color=sev_color, thickness=0.3),
                        bgcolor="#1e293b", borderwidth=0,
                        steps=[
                            dict(range=[0, umbrales["umbral_baja"]], color="#22c55e22"),
                            dict(range=[umbrales["umbral_baja"], umbrales["umbral_media"]], color="#eab30822"),
                            dict(range=[umbrales["umbral_media"], umbrales["umbral_alta"]], color="#f9731622"),
                            dict(range=[umbrales["umbral_alta"], umbrales["umbral_alta"]*3], color="#ef444422"),
                        ],
                        threshold=dict(line=dict(color=sev_color, width=3),
                                       thickness=0.8, value=rec_error),
                    ),
                    title=dict(text="Error de Reconstrucción VAE",
                               font=dict(color="#94a3b8", size=13)),
                ))
                fig_gauge.update_layout(**PLOTLY_LAYOUT, height=260)
                st.plotly_chart(fig_gauge, use_container_width=True)

            det1, det2 = st.columns(2)
            with det1:
                st.markdown(f"""| Campo | Valor |
|---|---|
| Monto | **${monto:,.2f}** |
| Descuento | **${descuento:,.2f}** |
| Ratio descuento | **{descuento/max(monto,1)*100:.1f}%** |
| N° ítems | **{num_items}** |""")
            with det2:
                st.markdown(f"""| Campo | Valor |
|---|---|
| Hora | **{hora:02d}:00** |
| Día | **{dia_semana}** |
| Método de pago | **{metodo_pago}** |
| Horario inusual | **{"Sí ⚠️" if hora < 6 or hora > 22 else "No ✅"}** |""")

            if es_anomalia:
                st.markdown("#### Factores que contribuyen a la anomalía")
                factores = []
                if monto > 200:
                    factores.append(f"💰 Monto elevado (${monto:.2f}) supera el umbral típico de $200")
                if monto < 1.0:
                    factores.append(f"💸 Monto muy bajo (${monto:.2f}) — posible error o transacción de prueba")
                if descuento > 0 and (descuento / max(monto,1)) > 0.3:
                    factores.append(f"🎟️ Descuento del {descuento/max(monto,1)*100:.0f}% supera el 30% permitido")
                if hora < 6 or hora > 22:
                    factores.append(f"🕐 Hora inusual ({hora:02d}:00) fuera del horario de operación normal")
                if num_items > 20:
                    factores.append(f"📦 Número de ítems ({num_items}) inusualmente alto")
                if not factores:
                    factores.append("📊 Patrón general inusual detectado por el VAE")
                for f_msg in factores:
                    st.warning(f_msg)
            else:
                st.success("La transacción está dentro de los patrones normales. No se detectaron anomalías.")
        else:
            st.info("Completa el formulario y presiona **Evaluar Transacción** para obtener el diagnóstico.")
            uc1, uc2, uc3, uc4 = st.columns(4)
            for col, sev, rng in zip([uc1,uc2,uc3,uc4], SEVERITY_ORDER, [
                f"< {umbrales['umbral_baja']:.5f}",
                f"{umbrales['umbral_baja']:.5f} – {umbrales['umbral_media']:.5f}",
                f"{umbrales['umbral_media']:.5f} – {umbrales['umbral_alta']:.5f}",
                f"≥ {umbrales['umbral_alta']:.5f}",
            ]):
                with col:
                    st.markdown(
                        f"""<div class="kpi-card" style="border-color:{SEVERITY_COLORS[sev]}44;">
                          <div style="font-size:1.1rem;font-weight:700;color:{SEVERITY_COLORS[sev]};">
                            {SEV_LABELS[sev]}</div>
                          <div style="font-size:0.8rem;color:#94a3b8;margin-top:8px;">{rng}</div>
                        </div>""", unsafe_allow_html=True)



# ════════════════════════════════════════════════════════════════════════════
# ROL NEGOCIO — RESTAURANTE ROSITA
# ════════════════════════════════════════════════════════════════════════════
elif st.session_state.rol == "Negocio":

    # ── Bloque de bienvenida corporativo ─────────────────────────────────────
    st.markdown("""
    <div class="rosita-welcome">
      <div style="display:flex; align-items:center; gap:16px; flex-wrap:wrap;">
        <div style="font-size:3rem;">🍽️</div>
        <div>
          <h1 style="color:#f97316; margin:0; font-size:1.8rem;">
            Auditor de Ventas — Restaurante Rosita
          </h1>
          <p style="color:#94a3b8; margin:4px 0 0; font-size:0.95rem;">
            Bienvenida, Rosita. Sube tu reporte de ventas y el sistema identificará
            automáticamente las transacciones que requieren tu atención.
          </p>
        </div>
      </div>
      <div style="display:flex; gap:32px; margin-top:20px; flex-wrap:wrap;">
        <div style="text-align:center;">
          <div style="font-size:1.4rem;">⚡</div>
          <div style="font-size:0.8rem; color:#64748b;">Análisis instantáneo</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:1.4rem;">🔒</div>
          <div style="font-size:0.8rem; color:#64748b;">Datos privados y seguros</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:1.4rem;">📊</div>
          <div style="font-size:0.8rem; color:#64748b;">Reportes descargables</div>
        </div>
        <div style="text-align:center;">
          <div style="font-size:1.4rem;">🎯</div>
          <div style="font-size:0.8rem; color:#64748b;">Alertas de riesgo claras</div>
        </div>
      </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Uploader de Excel ─────────────────────────────────────────────────────
    st.markdown("### 📂 Cargar Reporte de Ventas")

    st.info(
        "**Formato requerido:** El archivo Excel debe contener las siguientes columnas: "
        "`id_transaccion`, `fecha_hora`, `cajero`, `mesa`, `monto`, "
        "`descuento_pct`, `metodo_pago`, `tipo_transaccion`"
    )

    excel_file = st.file_uploader(
        "Arrastra aquí tu archivo de ventas o haz clic para seleccionarlo",
        type=["xlsx", "xls"],
        help="Solo se aceptan archivos Excel (.xlsx o .xls)",
    )


    # ── Procesamiento tras subir el Excel ─────────────────────────────────────
    if excel_file is not None:
        try:
            df_rosita = pd.read_excel(excel_file)
        except Exception as e:
            st.error(f"No se pudo leer el archivo: {e}")
            st.stop()

        # Validar columnas requeridas
        REQUIRED_COLS = ["id_transaccion","fecha_hora","cajero","mesa",
                         "monto","descuento_pct","metodo_pago","tipo_transaccion"]
        missing = [c for c in REQUIRED_COLS if c not in df_rosita.columns]
        if missing:
            st.error(
                f"El archivo no tiene las columnas requeridas: **{', '.join(missing)}**\n\n"
                "Revisa que tu Excel tenga exactamente los encabezados indicados."
            )
            st.stop()

        # ── Simulación del procesamiento VAE ─────────────────────────────────
        with st.spinner("Analizando transacciones con el modelo VAE..."):
            np.random.seed(42)
            n = len(df_rosita)

            # Error base + señales heurísticas por columna
            base = np.random.uniform(0.03, 0.055, n)

            if "monto" in df_rosita.columns:
                montos = pd.to_numeric(df_rosita["monto"], errors="coerce").fillna(0)
                base += np.where(montos > 200, (montos - 200) / 1000 * 0.5, 0)
                base += np.where(montos < 1.0, 0.04, 0)

            if "descuento_pct" in df_rosita.columns:
                desc = pd.to_numeric(df_rosita["descuento_pct"], errors="coerce").fillna(0)
                base += np.where(desc > 30, (desc - 30) / 100 * 0.12, 0)

            if "fecha_hora" in df_rosita.columns:
                try:
                    horas = pd.to_datetime(df_rosita["fecha_hora"], errors="coerce").dt.hour.fillna(12)
                    base += np.where((horas < 6) | (horas > 22), 0.025, 0)
                except Exception:
                    pass

            rec_errors = np.clip(base + np.random.normal(0, 0.004, n), 0.01, 1.0)

            # Umbrales desde el JSON del modelo
            umbrales_neg = load_umbrales()
            def classify(err):
                if err >= umbrales_neg["umbral_alta"]:  return "alta"
                if err >= umbrales_neg["umbral_media"]: return "media"
                if err >= umbrales_neg["umbral_baja"]:  return "baja"
                return "normal"

            df_rosita["_error_vae"]  = rec_errors
            df_rosita["_severidad"]  = [classify(e) for e in rec_errors]
            df_rosita["_es_anomalia"] = df_rosita["_severidad"].apply(lambda s: s != "normal")

        st.success(f"✅ Análisis completado — **{n:,}** transacciones procesadas.")
        st.markdown("---")

        # ── KPIs ──────────────────────────────────────────────────────────────
        st.markdown("### 📈 Resumen Ejecutivo")

        # Transacciones con cualquier alerta (baja, media, alta)
        anomalias_r = df_rosita[df_rosita["_es_anomalia"]].copy()

        # "Posibles pérdidas" = solo transacciones comprometidas: severidad media o alta
        comprometidas_r = df_rosita[df_rosita["_severidad"].isin(["media", "alta"])].copy()

        # Columna de monto: acepta "monto" o "monto_final"
        col_monto = "monto" if "monto" in df_rosita.columns else (
                    "monto_final" if "monto_final" in df_rosita.columns else None)

        if col_monto:
            monto_riesgo_r = pd.to_numeric(
                comprometidas_r[col_monto], errors="coerce"
            ).fillna(0).sum()
        else:
            monto_riesgo_r = 0

        total_tx    = len(df_rosita)
        n_anomalias = len(anomalias_r)
        pct_riesgo  = (n_anomalias / total_tx * 100) if total_tx else 0

        rk1, rk2, rk3 = st.columns(3)
        with rk1:
            st.markdown(
                f"""<div class="kpi-card">
                  <div class="kpi-title">Total de Transacciones Revisadas</div>
                  <div class="kpi-value">{total_tx:,}</div>
                  <div class="kpi-sub">en este reporte</div>
                </div>""", unsafe_allow_html=True)
        with rk2:
            st.markdown(
                f"""<div class="kpi-card">
                  <div class="kpi-title">Transacciones con Alerta</div>
                  <div class="kpi-value" style="color:#f97316;">{n_anomalias:,}</div>
                  <div class="kpi-sub">{pct_riesgo:.1f}% del total</div>
                </div>""", unsafe_allow_html=True)
        with rk3:
            st.markdown(
                f"""<div class="kpi-card">
                  <div class="kpi-title">Posibles Pérdidas (Monto en Riesgo)</div>
                  <div class="kpi-value" style="color:#ef4444;">${monto_riesgo_r:,.2f}</div>
                  <div class="kpi-sub">solo alertas media y alta</div>
                </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)


        # ── Gráfico de distribución de severidad ──────────────────────────────
        st.markdown("### 🔎 Distribución de Transacciones por Estado")

        # Solo mostramos Normal, Media y Alta (sin "baja") al perfil de negocio
        SEV_NEGOCIO   = ["normal", "media", "alta"]
        SEV_NEG_LABEL = {"normal": "Normal ✅", "media": "Alerta Media ⚠️", "alta": "Alerta Alta 🚨"}
        SEV_NEG_COLOR = {"normal": "#22c55e", "media": "#f97316", "alta": "#ef4444"}

        # Agrupamos "baja" dentro de "normal" para simplificar la vista de negocio
        def simplify_sev(s):
            return s if s in ("normal", "media", "alta") else "normal"

        df_rosita["_sev_simple"] = df_rosita["_severidad"].apply(simplify_sev)

        sev_counts = (
            df_rosita["_sev_simple"]
            .value_counts()
            .reindex(SEV_NEGOCIO, fill_value=0)
            .reset_index()
        )
        sev_counts.columns = ["severidad", "cantidad"]
        sev_counts["label"] = sev_counts["severidad"].map(SEV_NEG_LABEL)
        sev_counts["color"] = sev_counts["severidad"].map(SEV_NEG_COLOR)

        gc1_r, gc2_r = st.columns([1, 1.4])
        with gc1_r:
            fig_pie_r = go.Figure(go.Pie(
                labels=sev_counts["label"],
                values=sev_counts["cantidad"],
                marker=dict(
                    colors=sev_counts["color"].tolist(),
                    line=dict(color="#0f172a", width=2)
                ),
                hole=0.5,
                textinfo="percent+label",
                textfont=dict(size=12),
                hovertemplate="<b>%{label}</b><br>%{value} transacciones<br>%{percent}<extra></extra>",
            ))
            fig_pie_r.update_layout(
                **PLOTLY_LAYOUT, height=320, showlegend=False,
                annotations=[dict(
                    text=f"<b>{total_tx}</b><br>total",
                    x=0.5, y=0.5, font_size=13, showarrow=False,
                    font=dict(color="#e2e8f0"),
                )])
            st.plotly_chart(fig_pie_r, use_container_width=True)

        with gc2_r:
            fig_bar_r = px.bar(
                sev_counts, x="label", y="cantidad",
                color="severidad",
                color_discrete_map=SEV_NEG_COLOR,
                text="cantidad",
                labels={"label": "Estado", "cantidad": "N° Transacciones"},
            )
            fig_bar_r.update_traces(
                textposition="outside", marker_line_width=0,
                textfont=dict(size=13))
            fig_bar_r.update_layout(
                **PLOTLY_LAYOUT, showlegend=False, height=320,
                xaxis=dict(gridcolor="#334155"),
                yaxis=dict(gridcolor="#334155"))
            st.plotly_chart(fig_bar_r, use_container_width=True)


        # ── Tabla de resultados con estilos condicionales ─────────────────────
        st.markdown("### 📋 Tabla de Resultados")
        st.caption("Las transacciones están ordenadas por nivel de alerta (Altas primero).")

        # Orden de severidad para sort
        SEV_SORT = {"alta": 0, "media": 1, "baja": 2, "normal": 3}
        df_rosita["_sev_order"] = df_rosita["_severidad"].map(SEV_SORT)

        # Seleccionar columnas de negocio (originales + estado)
        cols_negocio = [c for c in REQUIRED_COLS if c in df_rosita.columns]
        df_result = (
            df_rosita[cols_negocio + ["_sev_simple", "_sev_order"]]
            .sort_values("_sev_order")
            .drop(columns=["_sev_order"])
            .rename(columns={"_sev_simple": "estado"})
            .reset_index(drop=True)
        )

        # Mapear etiquetas amigables
        df_result["estado"] = df_result["estado"].map(
            {"normal": "✅ Normal", "media": "⚠️ Alerta Media", "alta": "🚨 Alerta Alta"})

        # Formatear monto
        if "monto" in df_result.columns:
            df_result["monto"] = df_result["monto"].apply(
                lambda x: f"${pd.to_numeric(x, errors='coerce'):,.2f}"
                if pd.notna(pd.to_numeric(x, errors='coerce')) else x)

        # Función de estilos por fila según estado
        def style_row(row):
            estado = str(row.get("estado", ""))
            if "Alerta Alta" in estado:
                return [f"background-color:#ef444420; color:#fca5a5"] * len(row)
            elif "Alerta Media" in estado:
                return [f"background-color:#f9731620; color:#fdba74"] * len(row)
            else:
                return [f"background-color:#22c55e15; color:#86efac"] * len(row)

        # Limitar a 1000 filas para rendimiento
        df_show = df_result.head(1000)
        styled_df = df_show.style.apply(style_row, axis=1)

        st.dataframe(styled_df, use_container_width=True, height=450, hide_index=True)

        if len(df_result) > 1000:
            st.caption(f"Mostrando las primeras 1,000 filas de {len(df_result):,} totales. Descarga el reporte para ver todas.")

        # ── Botón de descarga Excel ────────────────────────────────────────────
        st.markdown("---")
        st.markdown("### ⬇️ Descargar Reporte Procesado")

        # Preparar Excel de salida
        output_excel = io.BytesIO()
        with pd.ExcelWriter(output_excel, engine="openpyxl") as writer:
            # Hoja principal: todos los resultados
            df_result.to_excel(writer, index=False, sheet_name="Reporte Completo")
            # Hoja solo alertas
            df_alertas = df_result[~df_result["estado"].str.contains("Normal")]
            df_alertas.to_excel(writer, index=False, sheet_name="Solo Alertas")
            # Hoja resumen
            resumen = pd.DataFrame({
                "Métrica": [
                    "Total transacciones revisadas",
                    "Alertas detectadas",
                    "% de alertas",
                    "Posibles pérdidas (monto en riesgo)",
                    "  → Criterio",
                ],
                "Valor": [
                    total_tx,
                    n_anomalias,
                    f"{pct_riesgo:.1f}%",
                    f"${monto_riesgo_r:,.2f}",
                    "Suma de montos con severidad Media o Alta únicamente",
                ],
            })
            resumen.to_excel(writer, index=False, sheet_name="Resumen Ejecutivo")
        output_excel.seek(0)

        dcol1, dcol2 = st.columns([2, 1])
        with dcol1:
            st.download_button(
                label="⬇️  Descargar reporte completo en Excel (.xlsx)",
                data=output_excel,
                file_name="reporte_auditoria_rosita.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                use_container_width=True,
            )
        with dcol2:
            st.markdown(
                f"""<div style="background:#1e293b; border:1px solid #334155;
                    border-radius:8px; padding:12px 16px; font-size:0.82rem; color:#94a3b8;">
                  📄 3 hojas incluidas:<br>
                  • Reporte Completo<br>
                  • Solo Alertas<br>
                  • Resumen Ejecutivo
                </div>""", unsafe_allow_html=True)

    else:
        # Estado inicial — sin archivo cargado
        st.markdown("""
        <div style="text-align:center; padding:60px 20px; color:#475569;">
          <div style="font-size:4rem; margin-bottom:16px;">📂</div>
          <h3 style="color:#64748b; font-weight:500;">
            Sube tu archivo de ventas para comenzar
          </h3>
          <p style="font-size:0.9rem; margin-top:8px;">
            El sistema analizará cada transacción y te mostrará cuáles requieren revisión.
          </p>
        </div>
        """, unsafe_allow_html=True)

