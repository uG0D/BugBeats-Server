import os
import joblib
import librosa
import numpy as np
import io
import wave
import soundfile as sf
import requests
from flask import Flask, request, jsonify

# --- CONFIGURACIÓN ---
MODEL_FILE = 'modelo_ratas.pkl'
UBIDOTS_TOKEN = "BBUS-05HpL3CGv101KvETp3hGXsPHSGQuJ6"
DEVICE_LABEL = "bugbeats"
VARIABLE_LABEL = "rata"

# UMBRALES DE SEGURIDAD (¡Aquí está la magia!)
MIN_VOLUME_RMS = 0.005  # Si el volumen es menor a esto, es Silencio absoluto.
MIN_CONFIDENCE = 0.65   # La IA debe estar 65% segura para dar la alerta.

app = Flask(__name__)
clf = None

# --- CARGA DEL MODELO ---
print("--- 🛡️ INICIANDO SERVIDOR CON FILTROS ---")
try:
    if os.path.exists(MODEL_FILE):
        clf = joblib.load(MODEL_FILE)
        print("✅ Cerebro cargado.")
    else:
        print(f"❌ ERROR: No encuentro '{MODEL_FILE}'")
except Exception as e:
    print(f"❌ ERROR CRÍTICO: {e}")

# Calentamiento (Para evitar Timeout)
def calentar_motores():
    try:
        dummy = np.zeros(16000)
        librosa.feature.mfcc(y=dummy, sr=16000, n_mfcc=13)
        print("🔥 Motores calientes.")
    except: pass

if clf: calentar_motores()

def convertir_raw_a_wav(raw_bytes, sample_rate=16000):
    try:
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(raw_bytes)
        wav_buffer.seek(0)
        return wav_buffer
    except Exception as e:
        print(f"Error WAV: {e}")
        return None

def enviar_a_ubidots(es_rata):
    # Enviamos en un bloque try/except silencioso para no detener el server
    try:
        val = 1.0 if es_rata else 0.0
        url = f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}"
        headers = {"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"}
        # Timeout de 3 segundos para no congelar la respuesta al Pico
        r = requests.post(url, json={VARIABLE_LABEL: val}, headers=headers, timeout=3)
        print(f"☁️ Ubidots Status Code: {r.status_code} | Valor: {val}")
    except Exception as e:
        print(f"⚠️ Ubidots falló: {e}")

@app.route('/', methods=['GET'])
def home():
    return "BugBeats Smart Server Active 🧠"

@app.route('/detectar', methods=['POST'])
def detectar():
    if clf is None: return jsonify({"error": "Modelo off"}), 500

    print(f"\n📞 Audio recibido: {len(request.data)} bytes")
    
    # 1. Convertir
    wav_file = convertir_raw_a_wav(request.data)
    if not wav_file: return jsonify({"error": "WAV falló"}), 400
    
    # 2. Leer y calcular Volumen (RMS)
    try:
        data, _ = sf.read(wav_file)
        if data.dtype != 'float32': data = data.astype('float32')
        
        # CÁLCULO DE VOLUMEN
        rms = np.sqrt(np.mean(data**2))
        print(f"🔊 Volumen detectado (RMS): {rms:.4f}")
        
        # FILTRO 1: EL PORTERO DE SILENCIO
        if rms < MIN_VOLUME_RMS:
            print("🛑 Audio demasiado bajo. Clasificado como SILENCIO/AMBIENTE.")
            enviar_a_ubidots(False)
            return jsonify({"status": "ok", "es_rata": 0, "mensaje": "SILENCIO 🔇", "prob": 0.0})

        # 3. Extraer características
        mfccs = librosa.feature.mfcc(y=data, sr=16000, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)

        # 4. Predicción con PROBABILIDAD
        # En lugar de .predict(), usamos .predict_proba()
        probs = clf.predict_proba([features])[0] # Devuelve [Prob_Ambiente, Prob_Rata]
        prob_rata = probs[1]
        
        print(f"📊 Confianza de IA -> Rata: {prob_rata*100:.1f}% | Ambiente: {probs[0]*100:.1f}%")

        # FILTRO 2: UMBRAL DE CONFIANZA
        es_rata = (prob_rata >= MIN_CONFIDENCE)
        
        msj = "RATA 🐀" if es_rata else "AMBIENTE 🍃"
        if es_rata: print("🚨 ¡ALERTA CONFIRMADA!")
        
        # Enviar respuesta
        enviar_a_ubidots(es_rata)
        
        return jsonify({
            "status": "ok", 
            "es_rata": 1 if es_rata else 0,
            "mensaje": msj,
            "confianza": float(prob_rata)
        })

    except Exception as e:
        print(f"❌ Error procesando: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
