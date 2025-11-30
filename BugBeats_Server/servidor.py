import os
import time
import threading

# --- CONFIGURACIÓN ANTI-CUELGUES ---
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['NUMBA_NUM_THREADS'] = '1'

import numpy as np
import librosa
import io
import soundfile as sf
import requests
from flask import Flask, request, jsonify

# Cargamos TF después de configurar entorno
import tensorflow as tf

MODEL_FILE = 'modelo_ratas.tflite'
UBIDOTS_TOKEN = "BBUS-05HpL3CGv101KvETp3hGXsPHSGQuJ6"
DEVICE_LABEL = "bugbeats"
VARIABLE_LABEL = "rata"
MIN_CONFIDENCE = 0.80 

app = Flask(__name__)

# Variables Globales del Cerebro
interpreter = None
input_details = None
output_details = None
model_ready = False

# --- CARGA DEL MODELO EN HILO APARTE (Para no bloquear el arranque) ---
def load_model_background():
    global interpreter, input_details, output_details, model_ready
    print("⏳ Cargando Inteligencia Artificial en segundo plano...")
    try:
        if os.path.exists(MODEL_FILE):
            interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
            interpreter.allocate_tensors()
            input_details = interpreter.get_input_details()
            output_details = interpreter.get_output_details()
            
            # Calentamiento (Hacer una predicción falsa para cargar librerías)
            print("🔥 Calentando motores...")
            dummy_input = np.zeros((1, 40), dtype=np.float32)
            interpreter.set_tensor(input_details[0]['index'], dummy_input)
            interpreter.invoke()
            
            model_ready = True
            print("✅ CEREBRO ONLINE Y LISTO.")
        else:
            print(f"❌ ERROR FATAL: No existe {MODEL_FILE}")
    except Exception as e:
        print(f"❌ Error cargando modelo: {e}")

# Iniciamos la carga inmediatamente
threading.Thread(target=load_model_background).start()

def procesar_audio(audio_data):
    # Normalizar
    max_val = np.max(np.abs(audio_data))
    if max_val > 0: audio_data = audio_data / max_val

    # Extraer MFCC (40 características)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    return np.array([features], dtype=np.float32)

def enviar_a_ubidots(es_rata):
    try:
        val = 1.0 if es_rata else 0.0
        requests.post(
            f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}",
            json={VARIABLE_LABEL: val},
            headers={"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"},
            timeout=1
        )
    except: pass

@app.route('/', methods=['GET'])
def home():
    status = "AI Ready 🧠" if model_ready else "AI Loading... ⏳"
    return f"BugBeats Server: {status}"

@app.route('/detectar', methods=['POST'])
def detectar():
    if not model_ready:
        return jsonify({"status": "error", "mensaje": "Servidor despertando, intenta en 10s..."}), 503

    print(f"\n📞 Audio recibido: {len(request.data)} bytes")
    
    try:
        # 1. Leer Audio (Soporta lo que mande el Pico)
        try:
            data, original_sr = sf.read(io.BytesIO(request.data))
        except:
            # Fallback raw
            return jsonify({"error": "Formato de audio corrupto"}), 400

        # 2. Resampling Inteligente (12k -> 16k)
        if original_sr != 16000:
            print(f"   🔄 Convirtiendo {original_sr}Hz -> 16000Hz")
            if len(data.shape) > 1: data = data.mean(axis=1) # Mono
            data = librosa.resample(data, orig_sr=original_sr, target_sr=16000)

        if data.dtype != 'float32': data = data.astype('float32')

        # 3. Inferencia Neuronal
        input_data = procesar_audio(data)
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prob_rata = float(output_data[0][0])
        
        print(f"🧠 Confianza: {prob_rata*100:.1f}%")

        es_rata = (prob_rata >= MIN_CONFIDENCE)
        enviar_a_ubidots(es_rata)
        
        return jsonify({
            "status": "ok", 
            "es_rata": 1 if es_rata else 0,
            "mensaje": "RATA 🐀" if es_rata else "AMBIENTE 🍃",
            "confianza": prob_rata
        })

    except Exception as e:
        print(f"❌ Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
