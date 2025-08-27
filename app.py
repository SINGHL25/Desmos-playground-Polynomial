import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
import joblib

st.set_page_config(page_title="Polynomial Playground 3D", layout="wide")
st.title("📊 Polynomial Playground – Multi-Curve & 3D Surface")

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.header("Settings")

# Multi-curve control
num_curves = st.sidebar.slider("Number of Curves", 1, 5, 1)
curve_names = [st.sidebar.text_input(f"Curve {i+1} Name", f"Curve_{i+1}") for i in range(num_curves)]

# Polynomial degrees per curve
degrees = [st.sidebar.selectbox(f"Degree {curve_names[i]}", [1,2,3,4,5], index=1) for i in range(num_curves)]

# 3D surface toggle
enable_3d = st.sidebar.checkbox("Enable 3D Bivariate Surface (z=f(x,y))")

# Save / Load
if st.sidebar.button("Save All Curves"):
    joblib.dump({"curve_names": curve_names, "degrees": degrees}, "saved_curves.pkl")
    st.sidebar.success("✅ Curves saved as `saved_curves.pkl`")

uploaded_file = st.sidebar.file_uploader("Load Saved Curves (.pkl)", type=["pkl"])
if uploaded_file:
    loaded = joblib.load(uploaded_file)
    curve_names = loaded.get("curve_names", curve_names)
    degrees = loaded.get("degrees", degrees)

# ----------------------------
# Curve Data Input
# ----------------------------
st.subheader("Curve Points Input")
curves_data = {}
for i in range(num_curves):
    st.markdown(f"### {curve_names[i]}")
    num_points = st.slider(f"Number of points for {curve_names[i]}", 3, 20, 5, key=i)
    x_min = st.number_input(f"X-min for {curve_names[i]}", value=-10.0, key=f"xmin{i}")
    x_max = st.number_input(f"X-max for {curve_names[i]}", value=10.0, key=f"xmax{i}")
    
    # Default points
    if 'x_points' not in st.session_state:
        st.session_state[f"x_points_{i}"] = list(np.linspace(x_min, x_max, num_points))
        st.session_state[f"y_points_{i}"] = [0]*num_points
    
    points_df = pd.DataFrame({
        "X": st.session_state[f"x_points_{i}"],
        "Y": st.session_state[f"y_points_{i}"]
    })
    edited_df = st.data_editor(points_df, num_rows="dynamic", key=f"editor{i}")
    st.session_state[f"x_points_{i}"] = edited_df["X"].values
    st.session_state[f"y_points_{i}"] = edited_df["Y"].values
    curves_data[curve_names[i]] = {
        "X": edited_df["X"].values,
        "Y": edited_df["Y"].values,
        "degree": degrees[i]
    }

# ----------------------------
# Plot Multiple Curves
# ----------------------------
fig = go.Figure()
for name, data in curves_data.items():
    poly = PolynomialFeatures(degree=data["degree"])
    X_poly = poly.fit_transform(data["X"].reshape(-1,1))
    model = LinearRegression()
    model.fit(X_poly, data["Y"])
    x_line = np.linspace(min(data["X"]), max(data["X"]), 500)
    y_line = model.predict(poly.transform(x_line.reshape(-1,1)))
    
    fig.add_trace(go.Scatter(x=data["X"], y=data["Y"], mode='markers', name=f"{name} Points"))
    fig.add_trace(go.Scatter(x=x_line, y=y_line, mode='lines', name=f"{name} Fit"))

fig.update_layout(
    title="Multiple Polynomial Curves",
    xaxis_title="X-axis",
    yaxis_title="Y-axis",
    width=900,
    height=600
)
st.plotly_chart(fig, use_container_width=True)

# ----------------------------
# Optional 3D Surface
# ----------------------------
if enable_3d:
    st.subheader("3D Bivariate Polynomial Surface (z = f(x, y))")
    x_range = np.linspace(-5,5,20)
    y_range = np.linspace(-5,5,20)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    
    # Example bivariate polynomial: z = 1 + 2x + 3y + xy + x^2 + y^2
    Z_grid = 1 + 2*X_grid + 3*Y_grid + X_grid*Y_grid + X_grid**2 + Y_grid**2
    
    fig3d = go.Figure(data=[go.Surface(z=Z_grid, x=X_grid, y=Y_grid)])
    fig3d.update_layout(
        title="3D Polynomial Surface",
        scene=dict(
            xaxis_title="X-axis",
            yaxis_title="Y-axis",
            zaxis_title="Z-axis"
        ),
        width=900,
        height=700
    )
    st.plotly_chart(fig3d, use_container_width=True)

# Footer
st.markdown("---")
st.markdown("💡 Tip: Drag points in the table or edit values to see curves update. Use the 3D toggle for multivariate polynomial surfaces.")
