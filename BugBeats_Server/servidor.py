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

# --- UMBRALES FÍSICOS (El filtro anti-estupidez) ---
MIN_VOLUME_RMS = 0.0001      # Umbral de silencio
MIN_CONFIDENCE = 0.65        # Confianza mínima de la IA
MIN_CENTROID_HZ = 1500       # ¡NUEVO! Si el sonido promedio es más grave que esto, NO es rata.
                             # Las ratas chillan entre 2000Hz y 8000Hz.
                             # La voz humana/musica suele estar por debajo de 1000Hz.

app = Flask(__name__)
clf = None

print("--- 🔬 SERVIDOR HÍBRIDO (FÍSICA + IA) ---")
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
        librosa.feature.spectral_centroid(y=dummy, sr=16000)
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
        # Timeout ultra corto (1s) para no bloquear
        requests.post(
            f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}",
            json={VARIABLE_LABEL: val},
            headers={"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"},
            timeout=1
        )
        print(f"☁️ Ubidots: {val}")
    except:
        print("⚠️ Ubidots falló (ignorado)")

@app.route('/', methods=['GET'])
def home():
    return "BugBeats Physics-Enhanced Server 🐀"

@app.route('/detectar', methods=['POST'])
def detectar():
    if clf is None: return jsonify({"error": "Modelo off"}), 500

    print(f"\n📞 Audio recibido: {len(request.data)} bytes")
    
    wav_file = convertir_raw_a_wav(request.data)
    if not wav_file: return jsonify({"error": "WAV falló"}), 400
    
    try:
        data, _ = sf.read(wav_file)
        if data.dtype != 'float32': data = data.astype('float32')
        
        # 1. ANÁLISIS DE VOLUMEN (RMS)
        rms_original = np.sqrt(np.mean(data**2))
        print(f"🔉 RMS Original: {rms_original:.6f}")

        # AMPLIFICACIÓN CONTROLADA
        max_val = np.max(np.abs(data))
        if max_val > 0:
            factor = 1.0 / max_val
            # Tope de x200. Más de eso es inventar datos sobre ruido.
            if factor > 200: factor = 200 
            data = data * factor
            print(f"🚀 Amplificado x{factor:.2f}")

        # 2. GUARDIÁN 1: SILENCIO
        if rms_original < MIN_VOLUME_RMS:
            print("🛑 Rechazo por Silencio Absoluto.")
            enviar_a_ubidots(False)
            return jsonify({"status": "ok", "es_rata": 0, "mensaje": "SILENCIO 🔇", "prob": 0.0})

        # 3. GUARDIÁN 2: FÍSICA DEL SONIDO (CENTROIDE ESPECTRAL)
        # Calculamos dónde está la "masa" del sonido (agudo vs grave)
        cent = librosa.feature.spectral_centroid(y=data, sr=16000)
        avg_centroid = np.mean(cent)
        print(f"🎼 Frecuencia Promedio: {avg_centroid:.1f} Hz")
        
        # SI EL SONIDO ES GRAVE (Musica, Voz, Golpes, Ruido estático grave) -> DESCARTAR
        if avg_centroid < MIN_CENTROID_HZ:
            print(f"🛑 Rechazo por Frecuencia Baja ({avg_centroid:.1f} Hz < {MIN_CENTROID_HZ} Hz). Es ruido o voz.")
            enviar_a_ubidots(False)
            return jsonify({"status": "ok", "es_rata": 0, "mensaje": "RUIDO GRAVE 📉", "prob": 0.0})

        # --- SI PASA LOS FILTROS, ENTRA LA IA ---
        mfccs = librosa.feature.mfcc(y=data, sr=16000, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)

        probs = clf.predict_proba([features])[0]
        prob_rata = probs[1]
        
        print(f"🧠 IA Opinión: {prob_rata*100:.1f}% Rata")

        # 4. GUARDIÁN 3: CONFIANZA
        es_rata = (prob_rata >= MIN_CONFIDENCE)
        
        enviar_a_ubidots(es_rata)
        
        msj = "RATA 🐀" if es_rata else "AMBIENTE 🍃"
        if es_rata: print("🚨 ¡CONFIRMADO!")
        
        return jsonify({
            "status": "ok", 
            "es_rata": 1 if es_rata else 0,
            "mensaje": msj,
            "confianza": float(prob_rata)
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
