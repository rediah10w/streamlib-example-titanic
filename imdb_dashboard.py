import streamlit as st
import pandas as pd

st.set_page_config(
    page_title="IMDB Movies Dashboard",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── Carga de datos ─────────────────────────────────────────────────────────────
URL = (
    "https://raw.githubusercontent.com/LearnDataSci/articles/refs/heads/master/"
    "Python%20Pandas%20Tutorial%20A%20Complete%20Introduction%20for%20Beginners/"
    "IMDB-Movie-Data.csv"
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv(URL)


@st.cache_data
def explode_genres(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["Genre"] = d["Genre"].str.split(",")
    d = d.explode("Genre")
    d["Genre"] = d["Genre"].str.strip()
    return d


df = load_data()
df_exploded = explode_genres(df)

# ── Encabezado ─────────────────────────────────────────────────────────────────
st.title("🎬 IMDB Movies Dashboard")
st.caption(
    f"Dataset: {len(df):,} películas  ·  "
    f"Años {int(df['Year'].min())}–{int(df['Year'].max())}  ·  "
    f"Rating promedio: {df['Rating'].mean():.2f}"
)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 1 — Desplegable por Género
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Exploración por Género")

all_genres = sorted(df_exploded["Genre"].dropna().unique())
default_idx = all_genres.index("Action") if "Action" in all_genres else 0
selected_genre = st.selectbox("Selecciona un género:", all_genres, index=default_idx)

df_genre = (
    df_exploded[df_exploded["Genre"] == selected_genre]
    .drop_duplicates(subset="Title")
    .sort_values("Rating", ascending=False)
    .reset_index(drop=True)
)
df_genre.index += 1  # empieza en 1

col_table, col_chart = st.columns([1.2, 1])

with col_table:
    st.subheader(f"Películas de {selected_genre}  ({len(df_genre)})")
    st.dataframe(
        df_genre[["Title", "Year", "Rating", "Director"]],
        use_container_width=True,
        height=420,
    )

with col_chart:
    st.subheader(f"Rating promedio por año — {selected_genre}")
    avg_by_year = (
        df_genre.groupby("Year")["Rating"]
        .mean()
        .round(2)
        .reset_index()
        .rename(columns={"Year": "Año", "Rating": "Rating promedio"})
    )
    avg_by_year["Color"] = "#F5C518"
    st.bar_chart(avg_by_year, x="Año", y="Rating promedio", color="Color", height=420)

# ══════════════════════════════════════════════════════════════════════════════
# SECCIÓN 2 — Slider por Año → Top 10
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.header("Top 10 películas mejor valoradas por año")

min_year = int(df["Year"].min())
max_year = int(df["Year"].max())

selected_year = st.slider(
    "Selecciona un año:",
    min_value=min_year,
    max_value=max_year,
    value=max_year,
)

top10 = (
    df[df["Year"] == selected_year]
    [["Title", "Genre", "Rating", "Director", "Revenue (Millions)"]]
    .sort_values("Rating", ascending=False)
    .head(10)
    .reset_index(drop=True)
)
top10.index += 1  # ranking 1–10

col_top, col_bar = st.columns([1.2, 1])

with col_top:
    st.subheader(f"Top 10 de {selected_year}")
    st.dataframe(top10, use_container_width=True, height=420)

with col_bar:
    st.subheader(f"Rating — Top 10 de {selected_year}")
    chart_data = top10[["Title", "Rating"]].copy()
    chart_data["Título"] = chart_data["Title"].str[:22] + "…"
    chart_data["Color"] = "#F5C518"
    st.bar_chart(chart_data, x="Título", y="Rating", color="Color", height=420)
