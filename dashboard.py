#Dependencias
import streamlit as st
import pandas as pd
from sklearn.metrics import precision_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier


#Iconos de la webapp
st.set_page_config(
    page_title="Que sabemos del Titanic?",
    layout="wide",
    initial_sidebar_state="collapsed",
)

#Carga de datos persistente en cache
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

#Metricas
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

#Graficos de barras y 
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
    for col, row in zip([pm1, pm2, pm3], emb_cnt.sort_values("Count", ascending=False).itertuples()):
        col.metric(row.Puerto, f"{row.Count:,}")


st.markdown("---")
c1, c2, c3 = st.columns(3)


with c1:
    st.subheader("¿Viajaba con Padres o Hijos?")
    parch_lbl = (df["Parch"] > 0).map({True: "Con padres/hijos", False: "Sin padres/hijos"})
    pc = parch_lbl.value_counts().reset_index()
    pc.columns = ["Condicion", "Count"]
    pc["Color"] = pc["Condicion"].map({"Con padres/hijos": "#4CAF50", "Sin padres/hijos": "#37474F"})
    st.bar_chart(pc, x="Condicion", y="Count", color="Color", height=410)

with c2:
    st.subheader("¿Viajaba con Hermanos o Pareja?")
    sibsp_lbl = (df["SibSp"] > 0).map({True: "Con hermanos/pareja", False: "Sin hermanos/pareja"})
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



#Arbol de decisiones para predecir supervivencia
@st.cache_resource
def prepare_model_data(df: pd.DataFrame) -> tuple[DecisionTreeClassifier, float, float, float]:
    data = df[["Survived", "Pclass", "Sex", "Age", "SibSp", "Parch"]].copy()
    data["Age"] = data["Age"].fillna(data["Age"].median())
    data["Sex"] = data["Sex"].map({"male": 0, "female": 1})

    X = data[["Pclass", "Sex", "Age", "SibSp", "Parch"]]
    y = data["Survived"]

    age_default = float(data["Age"].median())
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    model = DecisionTreeClassifier(random_state=42)
    model.fit(X_train, y_train)

    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)

    precision_train = precision_score(y_train, y_pred_train, zero_division=0)
    precision_test = precision_score(y_test, y_pred_test, zero_division=0)

    return model, age_default, precision_train, precision_test


#Formulario de prediccion

st.header("Prediccion de supervivencia")

model, age_default, precision_train, precision_test = prepare_model_data(df)
st.caption(f"Precision del arbol en entrenamiento: {precision_train:.3f}, precision en test: {precision_test:.3f}")

with st.form("prediction_form"):
    pclass_pred = st.selectbox("Clase", [1, 2, 3], format_func=lambda x: f"{x}a clase")
    sex_label = st.radio("Sexo", ["Hombre", "Mujer"], horizontal=True)
    age_pred = st.slider("Edad", min_value=0, max_value=80, value=int(age_default))
    sibsp_pred = st.number_input("Cantidad de Hermanos/Pareja", min_value=0, max_value=10, value=0, step=1)
    parch_pred = st.number_input("Cantidad de Padres/Hijos", min_value=0, max_value=10, value=0, step=1)
    submitted = st.form_submit_button("Predecir", type="primary")

if submitted:
    sex = 1 if sex_label == "Mujer" else 0
    sample = pd.DataFrame(
        [
            {
                "Pclass": int(pclass_pred),
                "Sex": sex,
                "Age": int(age_pred),
                "SibSp": int(sibsp_pred),
                "Parch": int(parch_pred),
            }
        ]
    )
    pred = int(model.predict(sample)[0])
    proba = float(model.predict_proba(sample)[0][1])

    if pred == 1:
        st.success(f"Prediccion: Sobrevive con {proba * 100:.1f}% de probabilidad")
    else:
        st.error(f"Prediccion: No sobrevive con {proba * 100:.1f}% de probabilidad de sobrevivir")

    #Preguntas
st.markdown("---")
st.subheader("Tabla de supervivencia por Sexo y Clase")
st.caption("Porcentaje de pasajeros que sobrevivieron según sexo y clase del tiquete")

_pivot = (
    df.pivot_table(values="Survived", index="Sex", columns="Pclass", aggfunc="mean")
    .mul(100)
    .round(1)
)
_pivot.index = _pivot.index.map({"male": "Hombre", "female": "Mujer"})
_pivot.columns = [f"{c}ª Clase" for c in _pivot.columns]
_pivot.index.name = "Sexo"
_pivot.columns.name = "Clase"

st.dataframe(
    _pivot.rename(columns=lambda c: c + "  (%)"),
    use_container_width=True,
)

st.markdown("---")
st.subheader("Pasajero que más pagó y no sobrevivió")

top = df[df["Survived"] == 0].sort_values("Fare", ascending=False).iloc[0]

t1, t2, t3, t4 = st.columns(4)
t1.metric("Nombre",  top["Name"])
t2.metric("Tarifa",  f"£ {top['Fare']:.2f}")
t3.metric("Clase",   f"{int(top['Pclass'])}ª Clase")
t4.metric("Edad",    f"{int(top['Age'])} años" if pd.notna(top["Age"]) else "Desconocida")




