import streamlit as st
from database import Database
import re

db = Database()

def is_valid_email(email):
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def is_valid_password(password):
    if len(password) < 6:
        return False, "Password must be at least 6 characters long"
    return True, ""

def show_login_page():
    st.markdown("""
        <style>
        .auth-title {
            text-align: center;
            color: #0071ce;
            font-size: 44px;  /* Enlarged text */
            font-weight: bold;
            margin-bottom: 2px;
        }
        .auth-subtitle {
            text-align: center;
            color: #666;
            font-size: 18px;
            margin-bottom: 32px;
        }
        </style>
    """, unsafe_allow_html=True)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        # Only the title and subtitle, no container
        st.markdown('<div class="auth-title">🛒 Walmart Sales</div>', unsafe_allow_html=True)
        st.markdown('<div class="auth-subtitle">Sales Forecasting System</div>', unsafe_allow_html=True)

        tab1, tab2 = st.tabs(["🔐 Login", "📝 Sign Up"])
        
        with tab1:
            st.markdown("### Welcome Back!")
            with st.form("login_form"):
                username = st.text_input("Username", placeholder="Enter your username")
                password = st.text_input("Password", type="password", placeholder="Enter your password")
                submit = st.form_submit_button("Login", use_container_width=True)
                
                if submit:
                    if not username or not password:
                        st.error("Please fill in all fields")
                    else:
                        success, user_data = db.verify_user(username, password)
                        if success:
                            st.session_state['authenticated'] = True
                            st.session_state['user'] = user_data
                            st.success("Login successful!")
                            st.rerun()
                        else:
                            st.error("Invalid username or password")
        
        with tab2:
            st.markdown("### Create New Account")
            with st.form("signup_form"):
                full_name = st.text_input("Full Name", placeholder="Enter your full name")
                email = st.text_input("Email", placeholder="Enter your email")
                new_username = st.text_input("Username", placeholder="Choose a username")
                new_password = st.text_input("Password", type="password", placeholder="Create a password")
                confirm_password = st.text_input("Confirm Password", type="password", placeholder="Confirm your password")
                
                signup = st.form_submit_button("Sign Up", use_container_width=True)
                
                if signup:
                    if not all([full_name, email, new_username, new_password, confirm_password]):
                        st.error("Please fill in all fields")
                    elif not is_valid_email(email):
                        st.error("Please enter a valid email address")
                    elif new_password != confirm_password:
                        st.error("Passwords do not match")
                    else:
                        valid, msg = is_valid_password(new_password)
                        if not valid:
                            st.error(msg)
                        else:
                            success, message = db.create_user(new_username, email, new_password, full_name)
                            if success:
                                st.success(message)
                                st.info("Please login with your credentials")
                            else:
                                st.error(message)

def logout():
    st.session_state['authenticated'] = False
    st.session_state['user'] = None
    st.rerun()

def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False
    
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    
    return st.session_state['authenticated']

def get_current_user():
    return st.session_state.get('user', None)
