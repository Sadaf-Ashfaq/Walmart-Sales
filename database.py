import sqlite3
import hashlib
from datetime import datetime

class Database:
    def __init__(self, db_name='walmart_users.db'):
        self.db_name = db_name
        self.init_database()
    
    def get_connection(self):
        conn = sqlite3.connect(self.db_name)
        conn.row_factory = sqlite3.Row
        return conn
    
    def init_database(self):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        # Create users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                email TEXT UNIQUE NOT NULL,
                password TEXT NOT NULL,
                full_name TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                last_login TIMESTAMP
            )
        ''')
        
        # Create user predictions table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id INTEGER NOT NULL,
                store INTEGER,
                department INTEGER,
                prediction_date DATE,
                lr_prediction REAL,
                rf_prediction REAL,
                avg_prediction REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                FOREIGN KEY (user_id) REFERENCES users (id)
            )
        ''')
        
        conn.commit()
        conn.close()
    
    def hash_password(self, password):
        return hashlib.sha256(password.encode()).hexdigest()
    
    def create_user(self, username, email, password, full_name):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            hashed_password = self.hash_password(password)
            
            cursor.execute('''
                INSERT INTO users (username, email, password, full_name)
                VALUES (?, ?, ?, ?)
            ''', (username, email, hashed_password, full_name))
            
            conn.commit()
            conn.close()
            return True, "Account created successfully!"
        except sqlite3.IntegrityError:
            return False, "Username or email already exists!"
        except Exception as e:
            return False, f"Error: {str(e)}"
    
    def verify_user(self, username, password):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            hashed_password = self.hash_password(password)
            
            cursor.execute('''
                SELECT * FROM users 
                WHERE username = ? AND password = ?
            ''', (username, hashed_password))
            
            user = cursor.fetchone()
            
            if user:
                # Update last login
                cursor.execute('''
                    UPDATE users 
                    SET last_login = ? 
                    WHERE username = ?
                ''', (datetime.now(), username))
                conn.commit()
                
                conn.close()
                return True, dict(user)
            else:
                conn.close()
                return False, None
        except Exception as e:
            return False, None
    
    def get_user_by_username(self, username):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('SELECT * FROM users WHERE username = ?', (username,))
        user = cursor.fetchone()
        
        conn.close()
        return dict(user) if user else None
    
    def save_prediction(self, user_id, store, dept, date, lr_pred, rf_pred, avg_pred):
        try:
            conn = self.get_connection()
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO predictions 
                (user_id, store, department, prediction_date, lr_prediction, rf_prediction, avg_prediction)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (user_id, store, dept, date, lr_pred, rf_pred, avg_pred))
            
            conn.commit()
            conn.close()
            return True
        except Exception as e:
            return False
    
    def get_user_predictions(self, user_id, limit=10):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT * FROM predictions 
            WHERE user_id = ? 
            ORDER BY created_at DESC 
            LIMIT ?
        ''', (user_id, limit))
        
        predictions = cursor.fetchall()
        conn.close()
        
        return [dict(pred) for pred in predictions]
    
    def get_user_stats(self, user_id):
        conn = self.get_connection()
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_predictions,
                AVG(avg_prediction) as avg_sales_predicted
            FROM predictions 
            WHERE user_id = ?
        ''', (user_id,))
        
        stats = cursor.fetchone()
        conn.close()
        
        return dict(stats) if stats else {'total_predictions': 0, 'avg_sales_predicted': 0}