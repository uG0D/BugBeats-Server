import os
import joblib
import librosa
import numpy as np
import io
import wave
import soundfile as sf  # Usaremos esto para leer más ligero
from flask import Flask, request, jsonify

# --- CONFIGURACIÓN ---
MODEL_FILE = 'modelo_ratas.pkl'
UBIDOTS_TOKEN = "BBUS-05HpL3CGv101KvETp3hGXsPHSGQuJ6"
DEVICE_LABEL = "bugbeats"
VARIABLE_LABEL = "rata"

app = Flask(__name__)
clf = None

# --- CARGA DEL MODELO ---
print("--- 🔵 INICIANDO SERVIDOR ---")
try:
    if os.path.exists(MODEL_FILE):
        clf = joblib.load(MODEL_FILE)
        print("✅ Cerebro cargado.")
    else:
        print(f"❌ ERROR: No encuentro '{MODEL_FILE}'")
except Exception as e:
    print(f"❌ ERROR CRÍTICO al cargar modelo: {e}")

# --- FUNCIÓN DE CALENTAMIENTO (CRUCIAL PARA RENDER) ---
# Esto obliga a librosa/numba a compilarse ANTES de recibir peticiones
def calentar_motores():
    print("🔥 Calentando motores de IA...")
    try:
        # Creamos 1 segundo de silencio falso
        dummy_audio = np.zeros(16000) 
        # Forzamos la ejecución de la función pesada
        librosa.feature.mfcc(y=dummy_audio, sr=16000, n_mfcc=13)
        print("✅ Motores listos y compilados. Esperando audio real.")
    except Exception as e:
        print(f"⚠️ Error en calentamiento (no fatal): {e}")

# Ejecutamos el calentamiento al iniciar el script
if clf is not None:
    calentar_motores()

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

def procesar_audio(wav_file_obj):
    try:
        # OPTIMIZACIÓN: Usar soundfile en lugar de librosa.load para ahorrar RAM
        # soundfile lee directo sin resampling (ya sabemos que viene en 16k)
        data, samplerate = sf.read(wav_file_obj)
        
        # Asegurar que sea float32 (lo que librosa necesita)
        if data.dtype != 'float32':
            data = data.astype('float32')

        mfccs = librosa.feature.mfcc(y=data, sr=16000, n_mfcc=13)
        features = np.mean(mfccs.T, axis=0)
        return features
    except Exception as e:
        print(f"❌ Error procesando: {e}")
        return None

def enviar_a_ubidots(es_rata):
    try:
        val = 1.0 if es_rata else 0.0
        # Timeout corto para no colgar el servidor
        requests.post(
            f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}",
            json={VARIABLE_LABEL: val},
            headers={"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"},
            timeout=5
        )
        print(f"✅ Ubidots: {val}")
    except Exception as e:
        print(f"⚠️ Ubidots falló (pero no importa): {e}")

@app.route('/', methods=['GET'])
def home():
    return "BugBeats AI Ready 🐀"

@app.route('/detectar', methods=['POST'])
def detectar():
    if clf is None: return jsonify({"error": "Modelo off"}), 500

    print(f"\n📞 Petición recibida ({len(request.data)} bytes)")
    
    wav_file = convertir_raw_a_wav(request.data)
    if not wav_file: return jsonify({"error": "WAV falló"}), 400
    
    features = procesar_audio(wav_file)
    if features is None: return jsonify({"error": "Procesamiento falló"}), 500
    
    prediccion = clf.predict([features])[0]
    es_rata = int(prediccion) == 1
    
    # Responder al Pico PRIMERO (para evitar Timeout -110)
    respuesta = jsonify({
        "status": "ok", 
        "es_rata": int(prediccion),
        "mensaje": "RATA 🐀" if es_rata else "AMBIENTE 🍃"
    })
    
    # Enviar a Ubidots DESPUÉS (o lanzar en hilo aparte idealmente, pero así vale)
    enviar_a_ubidots(es_rata)
    
    return respuesta

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
