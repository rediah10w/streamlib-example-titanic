import streamlit as st
import pandas as pd



st.set_page_config(
    page_title="Que sabemos del Titanic?",
    layout="wide",
    initial_sidebar_state="collapsed",
)


@st.cache_data
def load_data() -> pd.DataFrame:
    return pd.read_csv("titanic.csv")


df = load_data()
col_title, col_btn = st.columns([5, 1])
with col_title:
    st.title("¿Que sabemos del Titanic?")
with col_btn:
    st.write("")
    



st.dataframe(df.head(10))

total       = len(df)
surv_pct    = df["Survived"].mean() * 100
age_median  = df["Age"].median()
fare_median = df["Fare"].median()

m1, m2, m3, m4 = st.columns(4)
with m1:
    st.metric("Pasajeros muestreados", f"{total:,}")
with m2:
    st.metric("Sobrevivientes", f"{surv_pct:.3f} %")
with m3:
    st.metric("Mediana de edad", f"{age_median:.0f} años")
with m4:
    st.metric("Mediana de precio tiquete", f"$ {fare_median:.2f}")


c1, c2, c3 = st.columns(3)

with c1:
    st.subheader("Sexo de los Pasajeros")
    sex_cnt = df["Sex"].value_counts().reset_index()
    sex_cnt.columns = ["Sex", "Count"]
    sex_cnt["Label"] = sex_cnt["Sex"].map({"male": "Hombre", "female": "Mujer"})
    sex_cnt["Color"] = sex_cnt["Sex"].map({"male": "#4A90D9", "female": "#E91E8C"})
    st.bar_chart(sex_cnt, x="Label", y="Count", color="Color", height=390)

with c2:
    st.subheader("Supervivencia por edad")
    df_age = df.dropna(subset=["Age"]).copy()
    df_age["GrupoEdad"] = (df_age["Age"] // 5 * 5).astype(int)
    surv_age = (
        df_age.groupby("GrupoEdad")["Survived"]
        .mean()
        .mul(100)
        .round(1)
        .reset_index()
    )
    surv_age.columns = ["Edad", "Supervivencia (%)"]
    st.line_chart(surv_age, x="Edad", y="Supervivencia (%)", color="#4ECDC4", height=390)

with c3:
    st.subheader("Puerto de Embarque")
    emb_cnt = pd.DataFrame({
        "Code":   ["S",            "C",          "Q"],
        "Puerto": ["Southampton",  "Cherbourg",  "Queenstown"],
        "lat":    [50.9097,        49.6333,      51.8502],
        "lon":    [-1.4044,        -1.6167,      -8.2940],
        "color":  ["#FF6B35",      "#FFE66D",    "#4ECDC4"],
    })
    conteos = df["Embarked"].value_counts().rename("Count")
    emb_cnt["Count"] = emb_cnt["Code"].map(conteos)
    emb_cnt["size"]  =emb_cnt["Count"] * 100
    st.map(emb_cnt, latitude="lat", longitude="lon", color="color", size="size", zoom=4)
    pm1, pm2, pm3 = st.columns(3)
    for col, (_, row) in zip(
        [pm1, pm2, pm3],
        emb_cnt.sort_values("Count", ascending=False).iterrows(),
    ):
        col.metric(row["Puerto"], f"{int(row['Count']):,}")


st.markdown("---")
c1, c2, c3 = st.columns(3)


with c1:
    st.subheader("¿Viajaba con Padres o Hijos?")
    parch_lbl = df["Parch"].apply(lambda x: "Con padres/hijos" if x > 0 else "Sin padres/hijos")
    pc = parch_lbl.value_counts().reset_index()
    pc.columns = ["Condicion", "Count"]
    pc["Color"] = pc["Condicion"].map({"Con padres/hijos": "#4CAF50", "Sin padres/hijos": "#37474F"})
    st.bar_chart(pc, x="Condicion", y="Count", color="Color", height=410)

with c2:
    st.subheader("¿Viajaba con Hermanos o Pareja?")
    sibsp_lbl = df["SibSp"].apply(lambda x: "Con hermanos/pareja" if x > 0 else "Sin hermanos/pareja")
    sc = sibsp_lbl.value_counts().reset_index()
    sc.columns = ["Condicion", "Count"]
    sc["Color"] = sc["Condicion"].map({"Con hermanos/pareja": "#2196F3", "Sin hermanos/pareja": "#37474F"})
    st.bar_chart(sc, x="Condicion", y="Count", color="Color", height=410)

with c3:
    st.subheader("Clase del Tiquete")
    pcl = df["Pclass"].value_counts().sort_index().reset_index()
    pcl.columns = ["Clase", "Count"]
    pcl["Label"] = pcl["Clase"].map({1: "1a Clase", 2: "2a Clase", 3: "3a Clase"})
    pcl["Color"] = pcl["Label"].map({"1a Clase": "#FFD700", "2a Clase": "#B0B0B0", "3a Clase": "#CD7F32"})
    st.bar_chart(pcl, x="Label", y="Count", color="Color", height=410)



st.title("Saludos a los que estan viendo este mensaje")