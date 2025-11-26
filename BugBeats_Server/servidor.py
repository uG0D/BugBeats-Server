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

# Ajustamos el umbral de silencio para que sea más permisivo
MIN_VOLUME_RMS = 0.001  
MIN_CONFIDENCE = 0.60   

app = Flask(__name__)
clf = None

# --- CARGA DEL MODELO ---
print("--- 🚀 INICIANDO SERVIDOR CON BOOST DE AUDIO ---")
try:
    if os.path.exists(MODEL_FILE):
        clf = joblib.load(MODEL_FILE)
        print("✅ Cerebro cargado.")
    else:
        print(f"❌ ERROR: No encuentro '{MODEL_FILE}'")
except Exception as e:
    print(f"❌ ERROR CRÍTICO: {e}")

def calentar_motores():
    try:
        dummy = np.zeros(16000)
        librosa.feature.mfcc(y=dummy, sr=16000, n_mfcc=13)
        print("🔥 Motores listos.")
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
    try:
        val = 1.0 if es_rata else 0.0
        url = f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}"
        headers = {"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"}
        requests.post(url, json={VARIABLE_LABEL: val}, headers=headers, timeout=3)
        print(f"☁️ Ubidots actualizado: {val}")
    except Exception as e:
        print(f"⚠️ Ubidots error: {e}")

@app.route('/', methods=['GET'])
def home():
    return "BugBeats AI (Amplified) is Ready 🐀🔊"

@app.route('/detectar', methods=['POST'])
def detectar():
    if clf is None: return jsonify({"error": "Modelo off"}), 500

    print(f"\n📞 Audio recibido: {len(request.data)} bytes")
    
    wav_file = convertir_raw_a_wav(request.data)
    if not wav_file: return jsonify({"error": "WAV falló"}), 400
    
    try:
        # 1. Leer audio con soundfile (Ligero)
        data, _ = sf.read(wav_file)
        if data.dtype != 'float32': data = data.astype('float32')
        
        # 2. CALCULAR RMS ORIGINAL (Para logs)
        rms_original = np.sqrt(np.mean(data**2))
        print(f"🔉 Volumen Original (RMS): {rms_original:.5f}")

        # --- 3. NORMALIZACIÓN (EL SECRETO) ---
        # Si el audio es muy bajito, lo amplificamos digitalmente al máximo posible
        max_val = np.max(np.abs(data))
        if max_val > 0:
            factor_amplificacion = 1.0 / max_val
            data = data * factor_amplificacion
            print(f"🚀 Audio Amplificado x{factor_amplificacion:.2f} veces")
        
        # Nuevo RMS después de amplificar (solo por curiosidad)
        rms_boosted = np.sqrt(np.mean(data**2))
        
        # 4. FILTRO DE SILENCIO (Usamos el original para descartar ruido eléctrico puro)
        # Si incluso después de amplificar es puro ruido de fondo, lo matamos.
        # Pero ojo: comparamos el RMS original para no amplificar estática vacía.
        if rms_original < MIN_VOLUME_RMS:
            print("🛑 Señal muerta (Silencio absoluto).")
            enviar_a_ubidots(False)
            return jsonify({"status": "ok", "es_rata": 0, "mensaje": "SILENCIO 🔇", "prob": 0.0})

        # 5. IA entra en acción
        mfccs = librosa.feature.mfcc(y=data, sr=16000, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)

        probs = clf.predict_proba([features])[0]
        prob_rata = probs[1]
        
        print(f"📊 Confianza IA: Rata {prob_rata*100:.1f}% | Ambiente {probs[0]*100:.1f}%")

        es_rata = (prob_rata >= MIN_CONFIDENCE)
        
        # Respuesta
        enviar_a_ubidots(es_rata)
        
        return jsonify({
            "status": "ok", 
            "es_rata": 1 if es_rata else 0,
            "mensaje": "RATA 🐀" if es_rata else "AMBIENTE 🍃",
            "confianza": float(prob_rata)
        })

    except Exception as e:
        print(f"❌ Error procesando: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
