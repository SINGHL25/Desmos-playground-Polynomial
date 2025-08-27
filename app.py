import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

st.set_page_config(page_title="Polynomial Playground Regression", layout="wide")
st.title("📊 Polynomial Regression Playground – Interactive Data + Multi-Curve + 3D")

# ----------------------------
# Sidebar Settings
# ----------------------------
st.sidebar.header("Settings")

num_curves = st.sidebar.slider("Number of 1D Curves", 1, 5, 2)
degree = st.sidebar.slider("Polynomial Degree", 1, 5, 2)
num_points = st.sidebar.slider("Points per Curve", 3, 20, 5)
x_min = st.sidebar.number_input("X-min", value=-10.0)
x_max = st.sidebar.number_input("X-max", value=10.0)

# ----------------------------
# Data Upload / Manual Points
# ----------------------------
st.sidebar.header("Data Input")
data_option = st.sidebar.radio("Data Source", ["Random Generated", "Upload CSV", "Manual Points"])

if data_option == "Upload CSV":
    uploaded_file = st.sidebar.file_uploader("Upload CSV", type=["csv"])
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.write("### Uploaded Data Preview")
        st.dataframe(df.head())
        X_data = df.iloc[:,0].values.reshape(-1,1)
        Y_data = df.iloc[:,1].values
    else:
        st.stop()
elif data_option == "Manual Points":
    st.sidebar.write("Enter X and Y points separated by commas (e.g., -1,0,1,2)")
    x_input = st.sidebar.text_input("X values", "-3,-2,-1,0,1,2,3")
    y_input = st.sidebar.text_input("Y values", "-5,-2,-1,0,1,4,9")
    X_data = np.array([float(x) for x in x_input.split(",")]).reshape(-1,1)
    Y_data = np.array([float(y) for y in y_input.split(",")])
else:
    X_data = np.linspace(x_min, x_max, num_points).reshape(-1,1)
    Y_data = np.zeros_like(X_data)

# ----------------------------
# Plot Multiple Curves
# ----------------------------
st.sidebar.subheader("Curve Coefficients (Optional)")
X_plot = np.linspace(x_min, x_max, 500)
fig = go.Figure()
all_curves = []

for curve_idx in range(num_curves):
    coeffs = []
    for d in range(degree+1):
        coeff = st.sidebar.slider(f"Curve {curve_idx+1} a{d}", -10.0, 10.0, 1.0 if d==0 else 0.0, 0.1)
        coeffs.append(coeff)
    all_curves.append(coeffs)
    
    Y_plot = sum([coeffs[i]*X_plot**i for i in range(degree+1)])
    fig.add_trace(go.Scatter(x=X_plot.flatten(), y=Y_plot.flatten(), mode='lines', name=f"Curve {curve_idx+1}"))

    # Show equation
    equation_str = " + ".join([f"{coeffs[i]:.2f}*x^{i}" for i in range(degree+1)])
    st.sidebar.markdown(f"**Curve {curve_idx+1} Equation:** y = {equation_str}")

# ----------------------------
# Polynomial Regression Fitting
# ----------------------------
st.subheader("📈 Fit Polynomial Regression to Data")
deg_fit = st.slider("Degree of Regression Fit", 1, 5, 2)
poly_features = PolynomialFeatures(degree=deg_fit)
X_poly = poly_features.fit_transform(X_data)
model = LinearRegression()
model.fit(X_poly, Y_data)
Y_fit = model.predict(poly_features.transform(X_plot.reshape(-1,1)))
r2 = r2_score(Y_data, model.predict(X_poly))
st.write(f"**Fitted Equation:** y = { ' + '.join([f'{coef:.2f}*x^{i}' for i, coef in enumerate(model.coef_)]) } + {model.intercept_:.2f}")
st.write(f"**R² Score:** {r2:.3f}")

# Plot fitted regression and residuals
fig.add_trace(go.Scatter(x=X_plot.flatten(), y=Y_fit.flatten(), mode='lines', name="Fitted Regression", line=dict(color='black', dash='dash')))
fig.add_trace(go.Scatter(x=X_data.flatten(), y=Y_data.flatten(), mode='markers', name="Data Points", marker=dict(color='red', size=10)))

# ----------------------------
# Residuals Plot
# ----------------------------
residuals = Y_data - model.predict(X_poly)
st.subheader("Residuals")
st.bar_chart(residuals)

# ----------------------------
# Optional 3D Surface
# ----------------------------
enable_3d = st.sidebar.checkbox("Enable 3D Surface (z=f(x,y))")
if enable_3d:
    st.subheader("Interactive 3D Polynomial Surface")
    x_range = np.linspace(-5,5,20)
    y_range = np.linspace(-5,5,20)
    X_grid, Y_grid = np.meshgrid(x_range, y_range)
    st.sidebar.subheader("3D Surface Coefficients")
    c0 = st.sidebar.slider("c0", -5.0, 5.0, 1.0)
    c1 = st.sidebar.slider("c1*x", -5.0, 5.0, 1.0)
    c2 = st.sidebar.slider("c2*y", -5.0, 5.0, 1.0)
    c3 = st.sidebar.slider("c3*x*y", -5.0, 5.0, 0.5)
    c4 = st.sidebar.slider("c4*x^2", -5.0, 5.0, 0.5)
    c5 = st.sidebar.slider("c5*y^2", -5.0, 5.0, 0.5)
    Z_grid = c0 + c1*X_grid + c2*Y_grid + c3*X_grid*Y_grid + c4*X_grid**2 + c5*Y_grid**2
    fig3d = go.Figure(data=[go.Surface(z=Z_grid, x=X_grid, y=Y_grid)])
    fig3d.update_layout(title="3D Polynomial Surface", scene=dict(xaxis_title="X", yaxis_title="Y", zaxis_title="Z"), width=900, height=700)
    st.plotly_chart(fig3d, use_container_width=True)

# ----------------------------
# Display 1D Plot
# ----------------------------
fig.update_layout(title="Polynomial Playground - Multiple Curves + Regression Fit",
                  xaxis_title="X-axis", yaxis_title="Y-axis", width=900, height=600)
st.plotly_chart(fig, use_container_width=True)

st.markdown("---")
st.markdown("💡 Tip: Adjust coefficients, upload data, or manually enter points. Observe fitted curve and residuals interactively.")

