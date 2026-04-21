from flask import Flask, render_template, request, jsonify
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import joblib
import os
import io
import base64
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

app = Flask(__name__)

MODEL_PATH = "fire_model.pkl"
SCALER_PATH = "scaler.pkl"
ENCODERS_PATH = "encoders.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("Модель не найдена: fire_model.pkl")
    if not os.path.exists(SCALER_PATH):
        raise FileNotFoundError("Скалер не найден: scaler.pkl")
    if not os.path.exists(ENCODERS_PATH):
        raise FileNotFoundError("Кодировщики не найдены: encoders.pkl")
    
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    encoders = joblib.load(ENCODERS_PATH)
    
    return model, scaler, encoders

model, scaler, encoders = load_model()

def fig_to_base64(fig):
    buf = io.BytesIO()
    FigureCanvas(fig).print_png(buf)
    data = base64.b64encode(buf.getbuffer()).decode("ascii")
    plt.close(fig)
    return f"data:image/png;base64,{data}"

def get_danger_class(row):
    score = 0
    if row['FFMC'] > 85: score += 1
    if row['DMC'] > 40: score += 1
    if row['DC'] > 300: score += 1
    if row['ISI'] > 10: score += 1
    if row['temp'] > 25: score += 1
    if row['RH'] < 40: score += 1
    return min(score, 3)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json()
        
        month = data.get('month', 'aug')
        day = data.get('day', 'sun')
        temp = float(data.get('temp', 30.0))
        rh = float(data.get('rh', 25))
        wind = float(data.get('wind', 5.0))
        rain = float(data.get('rain', 0.0))
        ffmc = float(data.get('ffmc', 92.0))
        dmc = float(data.get('dmc', 120.0))
        dc = float(data.get('dc', 700.0))
        isi = float(data.get('isi', 15.0))
        
        try:
            month_enc = encoders['month'].transform([month])[0]
            day_enc = encoders['day'].transform([day])[0]
        except ValueError:
            month_enc, day_enc = 0, 0
            
        input_data = np.array([[month_enc, day_enc, ffmc, dmc, dc, isi, temp, rh, wind, rain]])
        input_scaled = scaler.transform(input_data)
        
        pred_class = int(model.predict(input_scaled)[0])
        proba = model.predict_proba(input_scaled)[0]
        confidence = float(proba[pred_class] * 100)
        
        class_names = ["Низкий", "Средний", "Высокий", "Экстремальный"]
        
        return jsonify({
            'success': True,
            'class': class_names[pred_class],
            'class_id': pred_class,
            'confidence': round(confidence, 1)
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        })

@app.route('/analytics', methods=['GET', 'POST'])
def analytics():
    plot_url = None
    stats = None
    error = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            error = "Файл не выбран"
        else:
            file = request.files['file']
            if file.filename == '':
                error = "Файл не выбран"
            else:
                try:
                    df = pd.read_csv(file)
                    
                    df['danger_class'] = df.apply(get_danger_class, axis=1)
                    
                    total_records = len(df)
                    avg_temp = df['temp'].mean()
                    high_danger = len(df[df['danger_class'] >= 2])
                    
                    stats = {
                        'total': total_records,
                        'avg_temp': round(avg_temp, 1),
                        'high_danger': high_danger
                    }
                    
                    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
                    
                    sns.boxplot(x='danger_class', y='temp', data=df, ax=ax1, color='lightgray')
                    ax1.axhline(y=25, color='black', linestyle='--', linewidth=1)
                    ax1.set_title('Температура по классам опасности')
                    ax1.set_xlabel('Класс опасности')
                    ax1.set_ylabel('Температура (C)')
                    
                    sns.boxplot(x='danger_class', y='RH', data=df, ax=ax2, color='lightgray')
                    ax2.axhline(y=40, color='black', linestyle='--', linewidth=1)
                    ax2.set_title('Влажность по классам опасности')
                    ax2.set_xlabel('Класс опасности')
                    ax2.set_ylabel('Влажность (%)')
                    
                    plot_url = fig_to_base64(fig)
                    
                except Exception as e:
                    error = f"Ошибка обработки файла: {str(e)}"
    
    return render_template('analytics.html', plot_url=plot_url, stats=stats, error=error)

@app.route('/report')
def report():
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/forest-fires/forestfires.csv"
    df_report = pd.read_csv(url)
    
    df_report['danger_class'] = df_report.apply(get_danger_class, axis=1)
    
    le_m = LabelEncoder()
    le_d = LabelEncoder()
    df_report['month_enc'] = le_m.fit_transform(df_report['month'])
    df_report['day_enc'] = le_d.fit_transform(df_report['day'])
    
    features = ['month_enc', 'day_enc', 'FFMC', 'DMC', 'DC', 'ISI', 'temp', 'RH', 'wind', 'rain']
    X = df_report[features]
    y = df_report['danger_class']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_test_scaled = scaler.transform(X_test)
    y_pred = model.predict(X_test_scaled)
    
    accuracy = accuracy_score(y_test, y_pred)
    
    report_dict = classification_report(y_test, y_pred, 
                                        target_names=['Низкий', 'Средний', 'Высокий', 'Экстремальный'],
                                        output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()
    report_df = report_df.round(3)
    report_html = report_df.to_html(classes='table')
    
    importance_df = pd.DataFrame({
        'Признак': features,
        'Важность': model.feature_importances_
    }).sort_values('Важность', ascending=False)
    importance_df['Важность'] = importance_df['Важность'].round(4)
    importance_html = importance_df.to_html(classes='table', index=False)
    
    cm = confusion_matrix(y_test, y_pred)
    fig_cm, ax_cm = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Greys',
                xticklabels=['Низкий', 'Средний', 'Высокий', 'Экстремальный'],
                yticklabels=['Низкий', 'Средний', 'Высокий', 'Экстремальный'],
                ax=ax_cm)
    ax_cm.set_xlabel('Предсказанный класс')
    ax_cm.set_ylabel('Истинный класс')
    cm_url = fig_to_base64(fig_cm)
    
    return render_template('report.html', 
                          accuracy=round(accuracy*100, 1),
                          report_table=report_html,
                          importance_table=importance_html,
                          cm_url=cm_url)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)