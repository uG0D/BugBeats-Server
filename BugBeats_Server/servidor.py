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

MIN_VOLUME_RMS = 0.0001
MIN_CONFIDENCE = 0.60

app = Flask(__name__)
clf = None

print("--- 🧠 SERVIDOR UNIVERSAL (WAV + RAW) ---")
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
    return "BugBeats Server Ready 🐀"

@app.route('/detectar', methods=['POST'])
def detectar():
    if clf is None: return jsonify({"error": "Modelo off"}), 500

    print(f"\n📞 Audio recibido: {len(request.data)} bytes")
    
    # --- CAMBIO CRÍTICO: DETECCIÓN AUTOMÁTICA DE FORMATO ---
    # Tu nuevo código envía un WAV real. El anterior enviaba RAW.
    # Este bloque maneja ambos casos.
    try:
        # Intento 1: Leer directo (Funciona si envías un WAV con encabezado)
        data, samplerate = sf.read(io.BytesIO(request.data))
        print("✅ Formato detectado: WAV Válido (desde archivo)")
    except:
        print("⚠️ No es WAV estándar. Intentando convertir desde RAW...")
        try:
            # Intento 2: Convertir raw a wav (Compatibilidad antigua)
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(request.data)
            wav_buffer.seek(0)
            data, samplerate = sf.read(wav_buffer)
        except Exception as e:
            return jsonify({"error": "Formato de audio no reconocido"}), 400

    # A partir de aquí, el proceso es igual
    try:
        if data.dtype != 'float32': data = data.astype('float32')
        
        rms_original = np.sqrt(np.mean(data**2))
        print(f"🔉 RMS: {rms_original:.6f}")

        # Amplificación
        max_val = np.max(np.abs(data))
        if max_val > 0:
            factor = 1.0 / max_val
            if factor > 500: factor = 500 
            data = data * factor
            print(f"🚀 Amplificado x{factor:.2f}")

        if rms_original < MIN_VOLUME_RMS:
            print("🛑 Silencio absoluto.")
            enviar_a_ubidots(False)
            return jsonify({"status": "ok", "es_rata": 0, "mensaje": "SILENCIO 🔇", "prob": 0.0})

        mfccs = librosa.feature.mfcc(y=data, sr=16000, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)

        probs = clf.predict_proba([features])[0]
        prob_rata = probs[1]
        
        print(f"🧠 IA: {prob_rata*100:.1f}% Rata")

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
