import sys
import os
import asyncio

# Parche de compatibilidad para Windows / Python
if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.set_page_config(page_title="Plataforma de Scouting & Predicción ML", layout="wide")

st.title("⚽ Plataforma Integral: Scouting, KPIs & Machine Learning")
st.markdown("Herramienta interactiva para análisis de futbolistas, comparación de mercado y predicción de resultados.")

# ======================================================
# 1. CARGA AUTOMÁTICA DIRECTA DESDE LA CARPETA
# ======================================================
origen_players = "players.csv" if os.path.exists("players.csv") else None
origen_games = "games.csv" if os.path.exists("games.csv") else None
archivo_appearances = None  # Carga directamente appearances_resumen.csv.gz o appearances.csv

# ======================================================
# 2. PROCESAMIENTO DE SCOUTING (PLAYERS & APPS RESUMEN)
# ======================================================
@st.cache_data
def cargar_y_procesar_scouting(fuente_players, fuente_appearances=None):
    df_players = pd.read_csv(fuente_players)
    
    # Normalización de nombre
    if 'name' not in df_players.columns and 'player_name' in df_players.columns:
        df_players['name'] = df_players['player_name']

    # Cálculo de edad
    if 'date_of_birth' in df_players.columns:
        df_players['date_of_birth'] = pd.to_datetime(df_players['date_of_birth'], errors='coerce')
        df_players['edad'] = 2026 - df_players['date_of_birth'].dt.year
    elif 'age' in df_players.columns:
        df_players['edad'] = df_players['age']
    else:
        df_players['edad'] = None

    # Normalización de valor de mercado en millones de euros
    if 'market_value_in_eur' in df_players.columns:
        df_players['valor_millones_eur'] = (pd.to_numeric(df_players['market_value_in_eur'], errors='coerce') / 1_000_000).round(2)
    else:
        df_players['valor_millones_eur'] = 0.0

    if 'height_in_cm' in df_players.columns:
        df_players['altura_cm'] = pd.to_numeric(df_players['height_in_cm'], errors='coerce')
    else:
        df_players['altura_cm'] = None

    stats = None

    # Prioridad 1: Archivo subido por el usuario en la sidebar
    if fuente_appearances is not None:
        try:
            df_temp = pd.read_csv(fuente_appearances)
            if 'partidos' in df_temp.columns and 'minutos' in df_temp.columns:
                stats = df_temp
            else:
                stats = df_temp.groupby('player_id').agg(
                    partidos=('minutes_played', 'count'),
                    minutos=('minutes_played', 'sum'),
                    goles=('goals', 'sum'),
                    asistencias=('assists', 'sum'),
                    amarillas=('yellow_cards', 'sum'),
                    rojas=('red_cards', 'sum')
                ).reset_index()
        except Exception:
            stats = None

    # Prioridad 2: Archivo ultraliviano comprimido en el directorio
    elif os.path.exists("appearances_resumen.csv.gz"):
        try:
            stats = pd.read_csv("appearances_resumen.csv.gz")
        except Exception:
            stats = None

    # Prioridad 3: Archivo tradicional appearances.csv
    elif os.path.exists("appearances.csv"):
        try:
            cols_app = ['player_id', 'minutes_played', 'goals', 'assists', 'yellow_cards', 'red_cards']
            df_app = pd.read_csv("appearances.csv", usecols=cols_app)
            stats = df_app.groupby('player_id').agg(
                partidos=('minutes_played', 'count'),
                minutos=('minutes_played', 'sum'),
                goles=('goals', 'sum'),
                asistencias=('assists', 'sum'),
                amarillas=('yellow_cards', 'sum'),
                rojas=('red_cards', 'sum')
            ).reset_index()
        except Exception:
            stats = None

    # Unión y cálculo de métricas avanzadas
    if stats is not None:
        if '90s' not in stats.columns and 'minutos' in stats.columns:
            stats['90s'] = stats['minutos'] / 90.0
            stats['goles_p90'] = np.where(stats['90s'] > 0, (stats['goles'] / stats['90s']).round(2), 0.0)
            stats['asist_p90'] = np.where(stats['90s'] > 0, (stats['asistencias'] / stats['90s']).round(2), 0.0)
            stats['goles_por_partido'] = np.where(stats['partidos'] > 0, (stats['goles'] / stats['partidos']).round(2), 0.0)
            stats['rojas_por_partido'] = np.where(stats['partidos'] > 0, (stats['rojas'] / stats['partidos']).round(3), 0.0)
            stats['amarillas_por_partido'] = np.where(stats['partidos'] > 0, (stats['amarillas'] / stats['partidos']).round(3), 0.0)

        df_completo = pd.merge(df_players, stats, on='player_id', how='left')
    else:
        df_completo = df_players

    # Reemplazo de nulos en métricas
    metricas_num = ['partidos', 'minutos', 'goles', 'asistencias', 'amarillas', 'rojas', 'goles_p90', 'asist_p90', 'goles_por_partido', 'rojas_por_partido', 'amarillas_por_partido']
    for m in metricas_num:
        if m in df_completo.columns:
            df_completo[m] = df_completo[m].fillna(0)
        else:
            df_completo[m] = 0

    return df_completo

if origen_players is None:
    st.warning("⚠️ Debes subir `players.csv` en la barra lateral o colocarlo en la carpeta del proyecto para comenzar.")
    st.stop()

with st.spinner("Cargando y procesando base de datos de jugadores..."):
    df = cargar_y_procesar_scouting(origen_players, archivo_appearances)

# ======================================================
# 3. ENTRENAMIENTO DEL MODELO ML CON IDENTIFICADOR DE LIGAS
# ======================================================
@st.cache_resource
def entrenar_modelo_prediccion(fuente_games):
    if fuente_games is None:
        return None, None, None, 0.0, None

    cols_games = ['game_id', 'date', 'competition_id', 'home_club_id', 'away_club_id', 
                  'home_club_name', 'away_club_name', 'home_club_goals', 'away_club_goals']
    try:
        games = pd.read_csv(fuente_games, usecols=lambda c: c in cols_games)
    except Exception:
        return None, None, None, 0.0, None

    games = games.dropna(subset=['home_club_goals', 'away_club_goals', 'home_club_name', 'away_club_name']).copy()
    games['date'] = pd.to_datetime(games['date'], errors='coerce')
    games = games.sort_values(by='date').reset_index(drop=True)

    if 'competition_id' not in games.columns:
        games['competition_id'] = "Otras Competiciones"

    # Mapeo amigable de códigos de ligas comunes
    nombre_ligas = {
        "ES1": "LaLiga (España)",
        "GB1": "Premier League (Inglaterra)",
        "IT1": "Serie A (Italia)",
        "L1": "Bundesliga (Alemania)",
        "FR1": "Ligue 1 (Francia)",
        "AR1N": "Primera División (Argentina)",
        "ARG1": "Primera División (Argentina)",
        "PO1": "Liga Portugal",
        "NL1": "Eredivisie (Países Bajos)",
        "CL": "UEFA Champions League"
    }
    games['liga_nombre'] = games['competition_id'].map(nombre_ligas).fillna(games['competition_id'])

    # Target: 1 = Gana Local, 0 = Empate, 2 = Gana Visitante
    condiciones = [
        games['home_club_goals'] > games['away_club_goals'],
        games['home_club_goals'] == games['away_club_goals'],
        games['home_club_goals'] < games['away_club_goals']
    ]
    valores = [1, 0, 2]
    games['target'] = np.select(condiciones, valores, default=0)

    # Historial de partidos
    registros = []
    for _, row in games.iterrows():
        pts_h = 3 if row['target'] == 1 else (1 if row['target'] == 0 else 0)
        registros.append({
            'date': row['date'], 'club': row['home_club_name'], 'liga': row['liga_nombre'],
            'gf': row['home_club_goals'], 'gc': row['away_club_goals'], 'pts': pts_h
        })
        pts_a = 3 if row['target'] == 2 else (1 if row['target'] == 0 else 0)
        registros.append({
            'date': row['date'], 'club': row['away_club_name'], 'liga': row['liga_nombre'],
            'gf': row['away_club_goals'], 'gc': row['home_club_goals'], 'pts': pts_a
        })

    df_hist = pd.DataFrame(registros).sort_values(by=['club', 'date']).reset_index(drop=True)
    
    # Forma reciente últimos 5 encuentros
    df_hist['pts_ultimos_5'] = df_hist.groupby('club')['pts'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df_hist['gf_ultimos_5'] = df_hist.groupby('club')['gf'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())
    df_hist['gc_ultimos_5'] = df_hist.groupby('club')['gc'].transform(lambda x: x.shift().rolling(5, min_periods=1).mean())

    metricas_club = df_hist[['date', 'club', 'pts_ultimos_5', 'gf_ultimos_5', 'gc_ultimos_5']].drop_duplicates(subset=['date', 'club'])
    
    games = games.merge(metricas_club, left_on=['date', 'home_club_name'], right_on=['date', 'club'], how='left')
    games = games.rename(columns={'pts_ultimos_5': 'h_pts_5', 'gf_ultimos_5': 'h_gf_5', 'gc_ultimos_5': 'h_gc_5'}).drop(columns=['club'])
    
    games = games.merge(metricas_club, left_on=['date', 'away_club_name'], right_on=['date', 'club'], how='left')
    games = games.rename(columns={'pts_ultimos_5': 'a_pts_5', 'gf_ultimos_5': 'a_gf_5', 'gc_ultimos_5': 'a_gc_5'}).drop(columns=['club'])

    features = ['h_pts_5', 'h_gf_5', 'h_gc_5', 'a_pts_5', 'a_gf_5', 'a_gc_5']
    df_ml = games.dropna(subset=features + ['target']).copy()

    X = df_ml[features]
    y = df_ml['target']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)
    modelo = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
    modelo.fit(X_train, y_train)

    precision = accuracy_score(y_test, modelo.predict(X_test))
    ultimo_estado = df_hist.sort_values(by='date').groupby('club').last().reset_index()

    return modelo, features, ultimo_estado, precision, games

# ======================================================
# 4. FILTROS GLOBALES EN SIDEBAR
# ======================================================
st.sidebar.subheader("🎯 Filtros de Scouting")

posiciones = sorted(df['position'].dropna().unique().tolist()) if 'position' in df.columns else []
posicion_sel = st.sidebar.multiselect("Posición:", posiciones)

min_edad = int(df['edad'].dropna().min()) if df['edad'].notna().any() else 16
max_edad = int(df['edad'].dropna().max()) if df['edad'].notna().any() else 42
rango_edad = st.sidebar.slider("Edad:", min_edad, max_edad, (min_edad, max_edad))

max_val = float(df['valor_millones_eur'].max()) if df['valor_millones_eur'].notna().any() else 100.0
presupuesto = st.sidebar.slider("Valor Máx (M€):", 0.0, max(max_val, 10.0), max_val)

min_partidos = st.sidebar.number_input("Mínimo de partidos jugados:", min_value=0, value=0, step=5)

df_filtrado = df.copy()
if posicion_sel:
    df_filtrado = df_filtrado[df_filtrado['position'].isin(posicion_sel)]
if df_filtrado['edad'].notna().any():
    df_filtrado = df_filtrado[(df_filtrado['edad'] >= rango_edad[0]) & (df_filtrado['edad'] <= rango_edad[1])]
df_filtrado = df_filtrado[df_filtrado['valor_millones_eur'] <= presupuesto]
df_filtrado = df_filtrado[df_filtrado['partidos'] >= min_partidos]

# ======================================================
# 5. PESTAÑAS PRINCIPALES DEL TABLERO
# ======================================================
tab_explorador, tab_detalle, tab_comparador, tab_ml = st.tabs([
    "📋 Base de Rendimiento",
    "👤 Ficha Técnica & KPIs",
    "👥 Comparador Multijugador (2 a 5)",
    "🤖 Predictor de Partidos & Fechas (ML)"
])

# --- PESTAÑA 1: EXPLORADOR GENERAL ---
with tab_explorador:
    st.subheader(f"Jugadores Filtrados ({len(df_filtrado)} registros)")
    cols_tabla = [
        'name', 'current_club_name', 'position', 'edad', 'partidos', 
        'goles', 'asistencias', 'goles_p90', 'goles_por_partido', 
        'rojas', 'valor_millones_eur'
    ]
    cols_existentes = [c for c in cols_tabla if c in df_filtrado.columns]
    st.dataframe(
        df_filtrado[cols_existentes].sort_values(by='valor_millones_eur', ascending=False).reset_index(drop=True),
        use_container_width=True
    )

# --- PESTAÑA 2: FICHA INDIVIDUAL CON KPIS ---
with tab_detalle:
    st.subheader("Perfil de Rendimiento Integral")
    jugadores_disp = sorted(df_filtrado['name'].dropna().unique().tolist())
    
    if jugadores_disp:
        jugador_sel = st.selectbox("Seleccionar jugador:", jugadores_disp)
        f = df[df['name'] == jugador_sel].iloc[0]
        
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Club Actual", str(f.get('current_club_name', 'N/D')))
        c2.metric("Edad", f"{int(f['edad'])} años" if pd.notna(f.get('edad')) else "N/D")
        c3.metric("Posición", str(f.get('position', 'N/D')))
        c4.metric("Valor Mercado", f"{f.get('valor_millones_eur', 0)} M€")
        
        st.write("---")
        st.markdown("#### ⚡ Métricas y Rendimiento")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Partidos Disputados", int(f.get('partidos', 0)))
        k2.metric("Minutos Jugados", f"{int(f.get('minutos', 0)):,}")
        k3.metric("Goles Totales", int(f.get('goles', 0)))
        k4.metric("Asistencias Totales", int(f.get('asistencias', 0)))

        r1, r2, r3, r4 = st.columns(4)
        r1.metric("Goles por Partido", f.get('goles_por_partido', 0))
        r2.metric("Goles cada 90' (p90)", f.get('goles_p90', 0))
        r3.metric("Amarillas / Partido", f.get('amarillas_por_partido', 0))
        r4.metric("Rojas / Partido", f.get('rojas_por_partido', 0))
    else:
        st.warning("No hay jugadores que coincidan con los filtros aplicados.")

# --- PESTAÑA 3: COMPARADOR MULTIJUGADOR (2 A 5) ---
with tab_comparador:
    st.subheader("Comparador Directo (Entre 2 y 5 futbolistas)")
    col_f1, col_f2 = st.columns(2)
    clubes_disponibles = sorted([c for c in df['current_club_name'].dropna().unique().tolist() if str(c).strip() != ""])
    
    with col_f1:
        club_filtro = st.multiselect("Filtrar por Club/Equipo:", options=clubes_disponibles, placeholder="Todos los clubes")
    with col_f2:
        posicion_filtro_comp = st.multiselect("Filtrar por Posición:", options=sorted(df['position'].dropna().unique().tolist()), placeholder="Todas las posiciones")
    
    df_pool = df.copy()
    if club_filtro:
        df_pool = df_pool[df_pool['current_club_name'].isin(club_filtro)]
    if posicion_filtro_comp:
        df_pool = df_pool[df_pool['position'].isin(posicion_filtro_comp)]
        
    df_pool_activos = df_pool[(df_pool['partidos'] > 0) | (df_pool['valor_millones_eur'] > 0)]
    lista_opciones = sorted(df_pool_activos['name'].dropna().unique().tolist())
    st.caption(f"Jugadores disponibles para seleccionar: **{len(lista_opciones)}**")
    st.write("---")

    c_sel1, c_sel2, c_sel3, c_sel4, c_sel5 = st.columns(5)
    opciones_con_vacio = [""] + lista_opciones
    with c_sel1:
        j1 = st.selectbox("Jugador 1:", opciones_con_vacio, index=1 if len(opciones_con_vacio) > 1 else 0, key="cmp_j1")
    with c_sel2:
        j2 = st.selectbox("Jugador 2:", opciones_con_vacio, index=2 if len(opciones_con_vacio) > 2 else 0, key="cmp_j2")
    with c_sel3:
        j3 = st.selectbox("Jugador 3:", opciones_con_vacio, index=0, key="cmp_j3")
    with c_sel4:
        j4 = st.selectbox("Jugador 4:", opciones_con_vacio, index=0, key="cmp_j4")
    with c_sel5:
        j5 = st.selectbox("Jugador 5:", opciones_con_vacio, index=0, key="cmp_j5")

    seleccionados = [j for j in [j1, j2, j3, j4, j5] if j != ""]
    if len(seleccionados) < 2:
        st.info("ℹ️ Selecciona al menos 2 jugadores para ver la comparativa.")
    else:
        df_comp = df[df['name'].isin(seleccionados)].copy()
        cols = st.columns(len(df_comp))
        for col, (_, row) in zip(cols, df_comp.iterrows()):
            with col:
                st.markdown(f"### {row['name']}")
                st.caption(f"{row.get('current_club_name', 'N/D')} | {row.get('position', 'N/D')}")
                st.metric("Goles / 90'", row.get('goles_p90', 0))
                st.metric("Goles / Partido", row.get('goles_por_partido', 0))
                st.metric("Asistencias / 90'", row.get('asist_p90', 0))
                st.metric("Rojas / Partido", row.get('rojas_por_partido', 0))
                st.metric("Valor Mercado", f"{row.get('valor_millones_eur', 0)} M€")
        st.write("---")
        
        kpi_elegido = st.selectbox("Selecciona KPI para el gráfico:", ["valor_millones_eur", "goles_p90", "goles_por_partido", "asist_p90", "rojas_por_partido", "partidos"], key="kpi_grafico_comp")
        fig_comp = px.bar(df_comp, x="name", y=kpi_elegido, color="name", text=kpi_elegido, title=f"Comparativa de {kpi_elegido.replace('_', ' ').upper()}", template="plotly_dark", height=400)
        fig_comp.update_layout(showlegend=False)
        st.plotly_chart(fig_comp, use_container_width=True)

# ------------------------------------------------------
# PESTAÑA 4: PREDICTOR MACHINE LEARNING CON FILTRO DE LIGAS
# ------------------------------------------------------
with tab_ml:
    st.subheader("Simulador y Predictor de Resultados (Por Liga y Próxima Fecha)")
    st.caption("Algoritmo Random Forest entrenado sobre la forma reciente (últimos 5 partidos) y localía.")

    with st.spinner("Entrenando algoritmo predictivo sobre games.csv..."):
        modelo, features, estado_equipos, precision, games_df = entrenar_modelo_prediccion(origen_games)

    if modelo is None:
        st.warning("⚠️ Sube `games.csv` en la barra lateral o colócalo en la carpeta para habilitar las predicciones.")
    else:
        st.sidebar.markdown("---")
        st.sidebar.markdown(f"🎯 **Precisión del Modelo (Test):** `{precision * 100:.1f}%`")

        def predecir_enfrentamiento(eq_local, eq_vis):
            h_data = estado_equipos[estado_equipos['club'] == eq_local]
            a_data = estado_equipos[estado_equipos['club'] == eq_vis]
            if h_data.empty or a_data.empty:
                return None
            
            s_h = h_data.iloc[0]
            s_a = a_data.iloc[0]

            v = pd.DataFrame([{
                'h_pts_5': s_h['pts_ultimos_5'], 'h_gf_5': s_h['gf_ultimos_5'], 'h_gc_5': s_h['gc_ultimos_5'],
                'a_pts_5': s_a['pts_ultimos_5'], 'a_gf_5': s_a['gf_ultimos_5'], 'a_gc_5': s_a['gc_ultimos_5']
            }])
            probs = modelo.predict_proba(v)[0]
            p_dict = dict(zip(modelo.classes_, probs))
            return {
                "local": p_dict.get(1, 0.0) * 100,
                "empate": p_dict.get(0, 0.0) * 100,
                "visita": p_dict.get(2, 0.0) * 100
            }

        ligas_disponibles = sorted(estado_equipos['liga'].dropna().unique().tolist())
        liga_seleccionada = st.selectbox("🏆 Selecciona la Liga o Torneo a simular:", ligas_disponibles)

        clubes_liga = sorted(estado_equipos[estado_equipos['liga'] == liga_seleccionada]['club'].unique().tolist())

        subtab_partido, subtab_fecha = st.tabs(["⚔️ Partido Individual", "📅 Simular Fecha Completa"])

        with subtab_partido:
            st.markdown(f"##### Simulación de Duelo Directo — {liga_seleccionada}")
            c_loc, c_vis = st.columns(2)
            with c_loc:
                eq_loc = st.selectbox("🏠 Equipo Local:", clubes_liga, index=0, key="ml_single_loc")
            with c_vis:
                idx_def_vis = 1 if len(clubes_liga) > 1 else 0
                eq_vis = st.selectbox("✈️ Equipo Visitante:", clubes_liga, index=idx_def_vis, key="ml_single_vis")

            if eq_loc == eq_vis:
                st.error("Selecciona dos equipos diferentes.")
            else:
                res = predecir_enfrentamiento(eq_loc, eq_vis)
                if res:
                    m1, m2, m3 = st.columns(3)
                    m1.metric(f"Victoria {eq_loc}", f"{res['local']:.1f}%")
                    m2.metric("Empate", f"{res['empate']:.1f}%")
                    m3.metric(f"Victoria {eq_vis}", f"{res['visita']:.1f}%")

                    df_plot = pd.DataFrame({
                        "Resultado": [f"Gana {eq_loc}", "Empate", f"Gana {eq_vis}"],
                        "Probabilidad (%)": [res['local'], res['empate'], res['visita']]
                    })
                    fig_p = px.bar(
                        df_plot, x="Resultado", y="Probabilidad (%)", color="Resultado",
                        text="Probabilidad (%)", color_discrete_sequence=["#2ecc71", "#f39c12", "#e74c3c"],
                        template="plotly_dark", height=350
                    )
                    fig_p.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
                    fig_p.update_layout(showlegend=False, yaxis=dict(range=[0, 100]))
                    st.plotly_chart(fig_p, use_container_width=True)

        with subtab_fecha:
            st.markdown(f"##### Cartelera y Pronóstico de la Próxima Jornada — {liga_seleccionada}")
            st.write("Genera una fecha automática cruzando los equipos de la liga o ajusta los duelos manualmente:")

            partidos_fecha = []
            for i in range(0, len(clubes_liga) - 1, 2):
                partidos_fecha.append((clubes_liga[i], clubes_liga[i+1]))

            resultados_fecha = []
            for loc, vis in partidos_fecha:
                pred = predecir_enfrentamiento(loc, vis)
                if pred:
                    max_prob = max(pred['local'], pred['empate'], pred['visita'])
                    if max_prob == pred['local']:
                        pronostico = f"1 (Gana {loc})"
                    elif max_prob == pred['empate']:
                        pronostico = "X (Empate)"
                    else:
                        pronostico = f"2 (Gana {vis})"

                    resultados_fecha.append({
                        "Partido": f"{loc} vs {vis}",
                        "% Local": f"{pred['local']:.1f}%",
                        "% Empate": f"{pred['empate']:.1f}%",
                        "% Visitante": f"{pred['visita']:.1f}%",
                        "Pronóstico ML": pronostico,
                        "Certeza": f"{max_prob:.1f}%"
                    })

            if resultados_fecha:
                df_fecha = pd.DataFrame(resultados_fecha)
                st.dataframe(df_fecha, use_container_width=True, hide_index=True)
            else:
                st.info("No hay suficientes clubes en esta categoría para armar una fecha completa.")