import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

st.title("NFL Win Prediction with XGBoost")

@st.cache_data
def load_data():
    nfl = pd.read_csv("season_2021.csv")
    nfl["result"] = (nfl["result"] == "W").astype(int)  # W = 1, T/L = 0
    return nfl

nfl = load_data()
features = nfl.iloc[:, 8:]  # all in-game stats
feature_names = features.columns

# --- Sidebar controls ---
st.sidebar.header("Options")
stat = st.sidebar.selectbox("Stat to explore (boxplot)", feature_names)
test_size = st.sidebar.slider("Test set size", 0.2, 0.5, 0.3, 0.05)

# --- 1. Boxplot of the selected stat by win/loss ---
st.subheader(f"Distribution of {stat} by game result")
fig, ax = plt.subplots()
sns.boxplot(data=nfl, x="result", y=stat, hue="result", legend=False, ax=ax)
ax.set_xticks([0, 1], ["Loss/Tie", "Win"])
st.pyplot(fig)

# --- 2. Train XGBoost ---
X_train, X_test, y_train, y_test = train_test_split(
    features, nfl["result"], test_size=test_size, random_state=42
)
model = xgb.XGBClassifier(n_estimators=100, learning_rate=0.1,
                          max_depth=3, random_state=42)
model.fit(X_train, y_train)
accuracy = accuracy_score(y_test, model.predict(X_test))

st.subheader("Model performance & feature importances")
st.metric("Test accuracy", f"{accuracy * 100:.1f}%")

fig, ax = plt.subplots()
sns.barplot(x=model.feature_importances_, y=feature_names,
            hue=feature_names, legend=False, ax=ax)
ax.set_xlabel("Importance (gain)")
st.pyplot(fig)

# --- 3. Predict the 2025 Seahawks season ---
st.subheader("2025 Seattle Seahawks: predicted vs actual wins")
new_data = pd.read_csv("seahawks_2025.csv")
new_preds = model.predict(new_data[feature_names])

comp = new_data[["week", "day", "date", "result", "opponent",
                 "tm_score", "opp_score"]].copy()
comp["predicted"] = new_preds
comp["actual"] = comp["result"]
st.dataframe(comp, width="stretch")
st.metric("Accuracy on 2025 data",
          f"{accuracy_score(comp['actual'], new_preds) * 100:.1f}%")
