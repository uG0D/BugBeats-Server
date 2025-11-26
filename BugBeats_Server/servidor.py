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

# Umbrales
MIN_VOLUME_RMS = 0.0001  # Muy sensible
MIN_CONFIDENCE = 0.65    # Confianza estándar

app = Flask(__name__)
clf = None

print("--- 🧠 SERVIDOR PURO (SOLO IA) ---")
try:
    if os.path.exists(MODEL_FILE):
        clf = joblib.load(MODEL_FILE)
        print("✅ Cerebro cargado.")
    else:
        print(f"❌ ERROR: No encuentro '{MODEL_FILE}'")
except Exception as e:
    print(f"❌ ERROR CRÍTICO: {e}")

# Calentamiento
if clf:
    try:
        dummy = np.zeros(16000)
        librosa.feature.mfcc(y=dummy, sr=16000, n_mfcc=13)
        print("🔥 Motores listos.")
    except: pass

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
    except: return None

def enviar_a_ubidots(es_rata):
    try:
        val = 1.0 if es_rata else 0.0
        requests.post(
            f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}",
            json={VARIABLE_LABEL: val},
            headers={"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"},
            timeout=1
        )
        print(f"☁️ Ubidots: {val}")
    except: pass

@app.route('/', methods=['GET'])
def home():
    return "BugBeats AI Server Ready 🐀"

@app.route('/detectar', methods=['POST'])
def detectar():
    if clf is None: return jsonify({"error": "Modelo off"}), 500

    print(f"\n📞 Audio recibido: {len(request.data)} bytes")
    
    wav_file = convertir_raw_a_wav(request.data)
    if not wav_file: return jsonify({"error": "WAV falló"}), 400
    
    try:
        data, _ = sf.read(wav_file)
        if data.dtype != 'float32': data = data.astype('float32')
        
        rms_original = np.sqrt(np.mean(data**2))
        print(f"🔉 RMS Original: {rms_original:.6f}")

        # AMPLIFICACIÓN
        max_val = np.max(np.abs(data))
        if max_val > 0:
            factor = 1.0 / max_val
            if factor > 500: factor = 500 
            data = data * factor
            print(f"🚀 Amplificado x{factor:.2f}")

        # FILTRO BASICO DE SILENCIO
        if rms_original < MIN_VOLUME_RMS:
            print("🛑 Silencio absoluto.")
            enviar_a_ubidots(False)
            return jsonify({"status": "ok", "es_rata": 0, "mensaje": "SILENCIO 🔇", "prob": 0.0})

        # IA DIRECTA
        mfccs = librosa.feature.mfcc(y=data, sr=16000, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)

        probs = clf.predict_proba([features])[0]
        prob_rata = probs[1]
        
        print(f"🧠 IA Dice: {prob_rata*100:.1f}% Rata")

        es_rata = (prob_rata >= MIN_CONFIDENCE)
        
        enviar_a_ubidots(es_rata)
        
        return jsonify({
            "status": "ok", 
            "es_rata": 1 if es_rata else 0,
            "mensaje": "RATA 🐀" if es_rata else "AMBIENTE 🍃",
            "confianza": float(prob_rata)
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)


