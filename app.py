import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from datetime import datetime, timedelta
import joblib
from PIL import Image

# Import authentication modules
try:
    from auth import check_authentication, show_login_page, logout, get_current_user
    from database import Database
    AUTH_ENABLED = True
    db = Database()
except ImportError:
    AUTH_ENABLED = False
    st.warning("⚠️ Authentication modules not found. Running without login.")

# Page configuration
st.set_page_config(
    page_title="Walmart Sales Forecasting",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS - Light Theme
st.markdown("""
    <style>
    .main {
        background-color: #ffffff;
    }
    .stButton>button {
        background-color: #0071ce;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 8px 20px;
        border: none;
        font-size: 14px;
    }
    .stButton>button:hover {
        background-color: #004f9a;
    }
    
    /* Custom metric boxes with different colors */
    .metric-box-blue {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
    }
    .metric-box-green {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
    }
    .metric-box-orange {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
    }
    .metric-box-purple {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        color: white;
    }
    
    .metric-label {
        font-size: 16px;
        font-weight: bold;
        margin-bottom: 10px;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .metric-value {
        font-size: 42px;
        font-weight: 900;
        margin: 5px 0;
    }
    .metric-subtitle {
        font-size: 14px;
        opacity: 0.9;
        font-weight: 600;
    }
    
    h1, h2, h3, h4, h5, h6 {
        color: #1a1a1a;
    }
    .info-box {
        background-color: #e3f2fd;
        padding: 15px;
        border-radius: 8px;
        border-left: 4px solid #0071ce;
        margin: 10px 0;
        color: #1a1a1a;
    }
    p, label, .stMarkdown {
        color: white;
    }
    </style>
""", unsafe_allow_html=True)

# Load models
@st.cache_resource
def load_models():
    try:
        lr_model = joblib.load('linear_regression_model.pkl')
        rf_model = joblib.load('random_forest_model.pkl')
        scaler = joblib.load('scaler.pkl')
        return lr_model, rf_model, scaler
    except:
        return None, None, None

# Load historical data
@st.cache_data
def load_data():
    try:
        df = pd.read_csv('predictions.csv')
        return df
    except:
        return None

# Check authentication - THIS IS THE KEY PART
if AUTH_ENABLED:
    if not check_authentication():
        show_login_page()
        st.stop()
    
    # Get current user
    current_user = get_current_user()
else:
    current_user = {'id': 1, 'username': 'guest', 'full_name': 'Guest User', 'email': 'guest@example.com', 'created_at': '', 'last_login': ''}

# Header
if AUTH_ENABLED:
    col1, col2, col3 = st.columns([2, 1, 1])
else:
    col1, col2, col3 = st.columns([3, 1, 0.5])

with col1:
    st.title("🛒 Walmart Sales Forecasting System")
    st.markdown("### AI-Powered Sales Prediction Dashboard")


if AUTH_ENABLED:
    with col3:
        st.markdown(f"**Welcome, {current_user['full_name']}!**")
        if st.button("🚪 Logout"):
            logout()

st.markdown("---")

# Load models and data
lr_model, rf_model, scaler = load_models()
historical_data = load_data()

# Sidebar
st.sidebar.header("📊 Navigation")

if AUTH_ENABLED:
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Make Prediction", "📈 Model Performance", "📊 Analytics", "👤 My Profile"])
else:
    page = st.sidebar.radio("Go to", ["🏠 Home", "🔮 Make Prediction", "📈 Model Performance", "📊 Analytics"])

st.sidebar.markdown("---")

# User stats in sidebar
if AUTH_ENABLED:
    user_stats = db.get_user_stats(current_user['id'])
    st.sidebar.markdown("### 📊 Your Stats")
    st.sidebar.metric("Total Predictions", user_stats['total_predictions'])
    if user_stats['avg_sales_predicted']:
        st.sidebar.metric("Avg Predicted Sales", f"${user_stats['avg_sales_predicted']:,.0f}")
    st.sidebar.markdown("---")

st.sidebar.info("""
**About This App**

This application uses machine learning models to forecast Walmart store sales based on historical data and various features.

**Models Used:**
- Linear Regression
- Random Forest

**Features:**
- Time-based patterns
- Store characteristics
- Economic indicators
- Historical trends
""")

# HOME PAGE
if page == "🏠 Home":
    st.header("Welcome to Walmart Sales Forecasting System")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("""
        <div class="metric-box-blue">
            <div class="metric-label">Total Stores</div>
            <div class="metric-value">45</div>
            <div class="metric-subtitle">Active</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-box-green">
            <div class="metric-label">Departments</div>
            <div class="metric-value">81</div>
            <div class="metric-subtitle">Tracked</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-box-orange">
            <div class="metric-label">Models</div>
            <div class="metric-value">2</div>
            <div class="metric-subtitle">ML Algorithms</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown("""
        <div class="metric-box-purple">
            <div class="metric-label">Accuracy</div>
            <div class="metric-value">95%</div>
            <div class="metric-subtitle">R² Score</div>
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🎯 Key Features")
        st.markdown("""
        - **Real-time Predictions**: Get instant sales forecasts
        - **Multiple Models**: Compare Linear Regression vs Random Forest
        - **Feature Engineering**: Utilizes lag features and rolling averages
        - **Visual Analytics**: Interactive charts and insights
        - **Easy to Use**: Simple interface for non-technical users
        """)
    
    with col2:
        st.subheader("🚀 How It Works")
        st.markdown("""
        1. **Input Store Data**: Enter store, department, and date information
        2. **Add Features**: Provide economic indicators and historical data
        3. **Generate Forecast**: Our ML models predict future sales
        4. **View Results**: See predictions with confidence intervals
        5. **Analyze Trends**: Explore interactive visualizations
        """)
    
    st.markdown("---")
    
    if historical_data is not None:
        st.subheader("📊 Recent Predictions Overview")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            y=historical_data['Actual_Sales'][:50],
            mode='lines',
            name='Actual Sales',
            line=dict(color='#0071ce', width=2)
        ))
        fig.add_trace(go.Scatter(
            y=historical_data['RF_Predicted'][:50],
            mode='lines',
            name='Predicted Sales',
            line=dict(color='#ff6b35', width=2, dash='dash')
        ))
        fig.update_layout(
            title='Actual vs Predicted Sales (Sample)',
            xaxis_title='Sample Index',
            yaxis_title='Weekly Sales ($)',
            height=400,
            hovermode='x unified',
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#1a1a1a')
        )
        st.plotly_chart(fig, use_container_width=True)

# PREDICTION PAGE
elif page == "🔮 Make Prediction":
    st.header("Make Sales Prediction")
    
    if lr_model is None or rf_model is None:
        st.error("⚠️ Models not found! Please ensure the model files are in the correct directory.")
        st.info("Required files: linear_regression_model.pkl, random_forest_model.pkl, scaler.pkl")
    else:
        st.markdown('<div class="info-box">Fill in the information below to generate a sales forecast.</div>', unsafe_allow_html=True)
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.subheader("🏪 Store Information")
            store = st.number_input("Store Number", min_value=1, max_value=45, value=1)
            dept = st.number_input("Department Number", min_value=1, max_value=99, value=1)
            store_type = st.selectbox("Store Type", ["A", "B", "C"])
        
        with col2:
            st.subheader("📅 Date Information")
            date = st.date_input("Select Date", datetime.now())
            year = date.year
            month = date.month
            week = date.isocalendar()[1]
            day = date.day
            day_of_week = date.weekday()
            quarter = (month - 1) // 3 + 1
        
        with col3:
            st.subheader("📊 Economic Indicators")
            temperature = st.slider("Temperature (°F)", 0.0, 100.0, 65.0)
            fuel_price = st.slider("Fuel Price ($/gallon)", 2.0, 5.0, 3.5, 0.1)
            cpi = st.slider("CPI", 100.0, 250.0, 180.0)
            unemployment = st.slider("Unemployment Rate (%)", 3.0, 15.0, 7.0, 0.1)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📈 Historical Sales Data")
            sales_lag_1 = st.number_input("Sales Last Week ($)", min_value=0.0, value=15000.0)
            sales_lag_2 = st.number_input("Sales 2 Weeks Ago ($)", min_value=0.0, value=14500.0)
            sales_lag_4 = st.number_input("Sales 4 Weeks Ago ($)", min_value=0.0, value=14000.0)
        
        with col2:
            st.subheader("📊 Rolling Averages")
            rolling_4 = st.number_input("4-Week Rolling Average ($)", min_value=0.0, value=14500.0)
            rolling_8 = st.number_input("8-Week Rolling Average ($)", min_value=0.0, value=14200.0)
        
        st.markdown("---")
        
        if st.button("🔮 Generate Forecast"):
            with st.spinner("Generating predictions..."):
                # Prepare input data
                input_data = pd.DataFrame({
                    'Store': [store],
                    'Dept': [dept],
                    'Year': [year],
                    'Month': [month],
                    'Week': [week],
                    'Day': [day],
                    'DayOfWeek': [day_of_week],
                    'Quarter': [quarter],
                    'Temperature': [temperature],
                    'Fuel_Price': [fuel_price],
                    'CPI': [cpi],
                    'Unemployment': [unemployment],
                    'Sales_Lag_1': [sales_lag_1],
                    'Sales_Lag_2': [sales_lag_2],
                    'Sales_Lag_4': [sales_lag_4],
                    'Sales_Rolling_Mean_4': [rolling_4],
                    'Sales_Rolling_Mean_8': [rolling_8]
                })
                
                # Scale features
                input_scaled = scaler.transform(input_data)
                
                # Make predictions
                lr_prediction = lr_model.predict(input_scaled)[0]
                rf_prediction = rf_model.predict(input_data)[0]
                avg_prediction = (lr_prediction + rf_prediction) / 2
                
                st.success("✅ Predictions Generated Successfully!")
                
                st.info("""
                **📊 Understanding Your Forecast:**
                
                Our system uses two machine learning models to predict sales:
                - **Linear Regression**: A simple model that finds linear relationships between features
                - **Random Forest**: An advanced model that uses multiple decision trees for better accuracy
                - **Average Prediction**: The ensemble average of both models, often providing the most reliable estimate
                
                The predictions are based on historical sales patterns, economic indicators (temperature, fuel prices, CPI, unemployment), 
                and store-specific characteristics. Higher values indicate expected strong sales performance for the selected date.
                """)
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown(f"""
                    <div class="metric-box-blue">
                        <div class="metric-label">Linear Regression</div>
                        <div class="metric-value">${lr_prediction:,.0f}</div>
                        <div class="metric-subtitle">ML Model 1</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.markdown(f"""
                    <div class="metric-box-green">
                        <div class="metric-label">Random Forest</div>
                        <div class="metric-value">${rf_prediction:,.0f}</div>
                        <div class="metric-subtitle">ML Model 2</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col3:
                    st.markdown(f"""
                    <div class="metric-box-purple">
                        <div class="metric-label">Average Prediction</div>
                        <div class="metric-value">${avg_prediction:,.0f}</div>
                        <div class="metric-subtitle">Ensemble</div>
                    </div>
                    """, unsafe_allow_html=True)
                
                st.markdown("<br>", unsafe_allow_html=True)
                st.markdown("---")
                
                # Visualization
                st.subheader("📊 Visual Comparison of Predictions")
                st.markdown("""
                **Graph Explanation:**
                - **X-Axis**: Shows the three prediction models (Linear Regression, Random Forest, and Average)
                - **Y-Axis**: Displays the predicted weekly sales amount in dollars ($)
                - Each colored bar represents the sales forecast from each model, making it easy to compare their predictions at a glance
                """)
                
                fig = go.Figure()
                
                models = ['Linear Regression', 'Random Forest', 'Average']
                predictions = [lr_prediction, rf_prediction, avg_prediction]
                colors = ['#667eea', '#11998e', '#4facfe']
                
                fig.add_trace(go.Bar(
                    x=models,
                    y=predictions,
                    marker_color=colors,
                    text=[f'${p:,.0f}' for p in predictions],
                    textposition='outside'
                ))
                
                fig.update_layout(
                    title='Sales Forecast Comparison',
                    yaxis_title='Predicted Sales ($)',
                    height=500,
                    showlegend=False,
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    font=dict(color='#1a1a1a')
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
                # Save prediction to database
                if AUTH_ENABLED:
                    db.save_prediction(
                        current_user['id'],
                        store,
                        dept,
                        date,
                        lr_prediction,
                        rf_prediction,
                        avg_prediction
                    )
                
                # Download results
                result_df = pd.DataFrame({
                    'Store': [store],
                    'Department': [dept],
                    'Date': [date],
                    'Linear_Regression_Prediction': [lr_prediction],
                    'Random_Forest_Prediction': [rf_prediction],
                    'Average_Prediction': [avg_prediction]
                })
                
                csv = result_df.to_csv(index=False)
                st.download_button(
                    label="📥 Download Prediction Results",
                    data=csv,
                    file_name=f'sales_prediction_{date}.csv',
                    mime='text/csv'
                )

# MODEL PERFORMANCE PAGE
elif page == "📈 Model Performance":
    st.header("Model Performance Metrics")
    
    if historical_data is not None:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        
        # Calculate metrics
        lr_mse = mean_squared_error(historical_data['Actual_Sales'], historical_data['LR_Predicted'])
        lr_rmse = np.sqrt(lr_mse)
        lr_mae = mean_absolute_error(historical_data['Actual_Sales'], historical_data['LR_Predicted'])
        lr_r2 = r2_score(historical_data['Actual_Sales'], historical_data['LR_Predicted'])
        
        rf_mse = mean_squared_error(historical_data['Actual_Sales'], historical_data['RF_Predicted'])
        rf_rmse = np.sqrt(rf_mse)
        rf_mae = mean_absolute_error(historical_data['Actual_Sales'], historical_data['RF_Predicted'])
        rf_r2 = r2_score(historical_data['Actual_Sales'], historical_data['RF_Predicted'])
        
        st.subheader("📊 Performance Comparison")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("### Linear Regression")
            st.markdown(f"""
            <div class="metric-box-blue">
                <div style="margin: 10px 0;">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value" style="font-size: 32px;">${lr_rmse:,.2f}</div>
                </div>
                <div style="margin: 10px 0;">
                    <div class="metric-label">MAE</div>
                    <div class="metric-value" style="font-size: 32px;">${lr_mae:,.2f}</div>
                </div>
                <div style="margin: 10px 0;">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-value" style="font-size: 32px;">{lr_r2:.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown("### Random Forest")
            st.markdown(f"""
            <div class="metric-box-green">
                <div style="margin: 10px 0;">
                    <div class="metric-label">RMSE</div>
                    <div class="metric-value" style="font-size: 32px;">${rf_rmse:,.2f}</div>
                </div>
                <div style="margin: 10px 0;">
                    <div class="metric-label">MAE</div>
                    <div class="metric-value" style="font-size: 32px;">${rf_mae:,.2f}</div>
                </div>
                <div style="margin: 10px 0;">
                    <div class="metric-label">R² Score</div>
                    <div class="metric-value" style="font-size: 32px;">{rf_r2:.4f}</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("---")
        
        # Comparison table
        comparison_df = pd.DataFrame({
            'Metric': ['RMSE', 'MAE', 'R² Score'],
            'Linear Regression': [f'${lr_rmse:,.2f}', f'${lr_mae:,.2f}', f'{lr_r2:.4f}'],
            'Random Forest': [f'${rf_rmse:,.2f}', f'${rf_mae:,.2f}', f'{rf_r2:.4f}']
        })
        
        st.subheader("📋 Detailed Metrics")
        st.dataframe(comparison_df, use_container_width=True)
        
        st.markdown("---")
        
        # Residual plots
        st.subheader("📉 Residual Analysis")
        
        col1, col2 = st.columns(2)
        
        with col1:
            lr_residuals = historical_data['Actual_Sales'] - historical_data['LR_Predicted']
            fig = px.scatter(x=historical_data['LR_Predicted'], y=lr_residuals,
                           labels={'x': 'Predicted Sales', 'y': 'Residuals'},
                           title='Linear Regression Residuals')
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            rf_residuals = historical_data['Actual_Sales'] - historical_data['RF_Predicted']
            fig = px.scatter(x=historical_data['RF_Predicted'], y=rf_residuals,
                           labels={'x': 'Predicted Sales', 'y': 'Residuals'},
                           title='Random Forest Residuals')
            fig.add_hline(y=0, line_dash="dash", line_color="red")
            fig.update_layout(plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
    else:
        st.warning("⚠️ Historical data not found. Please run the model training first.")

# ANALYTICS PAGE
elif page == "📊 Analytics":
    st.header("Sales Analytics Dashboard")
    
    if historical_data is not None:
        st.subheader("📈 Prediction Accuracy Over Time")
        
        # Scatter plot
        fig = go.Figure()
        
        fig.add_trace(go.Scatter(
            x=historical_data['Actual_Sales'],
            y=historical_data['RF_Predicted'],
            mode='markers',
            name='Random Forest',
            marker=dict(color='#11998e', size=8, opacity=0.6)
        ))
        
        # Perfect prediction line
        min_val = min(historical_data['Actual_Sales'].min(), historical_data['RF_Predicted'].min())
        max_val = max(historical_data['Actual_Sales'].max(), historical_data['RF_Predicted'].max())
        fig.add_trace(go.Scatter(
            x=[min_val, max_val],
            y=[min_val, max_val],
            mode='lines',
            name='Perfect Prediction',
            line=dict(color='red', dash='dash')
        ))
        
        fig.update_layout(
            title='Actual vs Predicted Sales',
            xaxis_title='Actual Sales ($)',
            yaxis_title='Predicted Sales ($)',
            height=500,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#1a1a1a')
        )
        
        st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Prediction Distribution")
            fig = go.Figure()
            fig.add_trace(go.Histogram(x=historical_data['Actual_Sales'], name='Actual', opacity=0.7))
            fig.add_trace(go.Histogram(x=historical_data['RF_Predicted'], name='Predicted', opacity=0.7))
            fig.update_layout(barmode='overlay', height=400, plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.subheader("📉 Error Distribution")
            errors = historical_data['Actual_Sales'] - historical_data['RF_Predicted']
            fig = px.histogram(errors, nbins=50, title='Prediction Errors')
            fig.update_layout(height=400, plot_bgcolor='white', paper_bgcolor='white', font=dict(color='#1a1a1a'))
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Statistics
        st.subheader("📋 Statistical Summary")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-box-blue">
                <div class="metric-label">Mean Actual Sales</div>
                <div class="metric-value" style="font-size: 36px;">${historical_data['Actual_Sales'].mean():,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            st.markdown(f"""
            <div class="metric-box-green">
                <div class="metric-label">Mean Predicted Sales</div>
                <div class="metric-value" style="font-size: 36px;">${historical_data['RF_Predicted'].mean():,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            mean_error = (historical_data['Actual_Sales'] - historical_data['RF_Predicted']).mean()
            st.markdown(f"""
            <div class="metric-box-orange">
                <div class="metric-label">Mean Prediction Error</div>
                <div class="metric-value" style="font-size: 36px;">${mean_error:,.0f}</div>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.warning("⚠️ No data available for analytics. Please run the model training first.")

# MY PROFILE PAGE (only if auth is enabled)
elif AUTH_ENABLED and page == "👤 My Profile":
    st.header("My Profile")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown(f"""
        <div class="metric-box-blue">
            <div class="metric-label">User Profile</div>
            <div style="margin: 20px 0;">
                <h2 style="color: white; margin: 10px 0;">👤</h2>
                <p style="font-size: 20px; font-weight: bold; margin: 5px 0;">{current_user['full_name']}</p>
                <p style="font-size: 14px; opacity: 0.9;">@{current_user['username']}</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        user_stats = db.get_user_stats(current_user['id'])
        st.markdown(f"""
        <div class="metric-box-green">
            <div class="metric-label">Total Predictions</div>
            <div class="metric-value">{user_stats['total_predictions']}</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.subheader("📋 Account Information")
        
        info_df = pd.DataFrame({
            'Field': ['Full Name', 'Username', 'Email', 'Member Since', 'Last Login'],
            'Value': [
                current_user['full_name'],
                current_user['username'],
                current_user['email'],
                current_user['created_at'].split()[0] if current_user['created_at'] else 'N/A',
                current_user['last_login'].split()[0] if current_user['last_login'] else 'N/A'
            ]
        })
        
        st.dataframe(info_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    st.subheader("📊 Recent Predictions History")
    
    recent_predictions = db.get_user_predictions(current_user['id'], limit=10)
    
    if recent_predictions:
        pred_df = pd.DataFrame(recent_predictions)
        
        # Format the dataframe
        display_df = pd.DataFrame({
            'Date': [p['prediction_date'] for p in recent_predictions],
            'Store': [p['store'] for p in recent_predictions],
            'Department': [p['department'] for p in recent_predictions],
            'LR Prediction': [f"${p['lr_prediction']:,.2f}" for p in recent_predictions],
            'RF Prediction': [f"${p['rf_prediction']:,.2f}" for p in recent_predictions],
            'Avg Prediction': [f"${p['avg_prediction']:,.2f}" for p in recent_predictions],
            'Created': [p['created_at'].split()[0] for p in recent_predictions]
        })
        
        st.dataframe(display_df, use_container_width=True, hide_index=True)
        
        # Visualization of predictions
        st.markdown("---")
        st.subheader("📈 Your Prediction Trends")
        
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=list(range(len(recent_predictions))),
            y=[p['avg_prediction'] for p in recent_predictions],
            mode='lines+markers',
            name='Average Predictions',
            line=dict(color='#0071ce', width=3),
            marker=dict(size=10)
        ))
        
        fig.update_layout(
            title='Your Recent Sales Predictions',
            xaxis_title='Prediction Number',
            yaxis_title='Predicted Sales ($)',
            height=400,
            plot_bgcolor='white',
            paper_bgcolor='white',
            font=dict(color='#1a1a1a'),
            hovermode='x unified'
        )
        
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info("No predictions yet. Make your first prediction to see your history!")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #666;'>
    <p>Walmart Sales Forecasting System | Powered by Machine Learning</p>
    <p>© 2025 - Built with Streamlit</p>
</div>
""", unsafe_allow_html=True)