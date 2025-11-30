import os

# --- FIX CRÍTICO PARA RENDER ---
# Configuramos las carpetas de caché en /tmp ANTES de importar librosa.
# Esto evita que Numba/Librosa se cuelguen intentando escribir donde no deben.
os.environ['NUMBA_CACHE_DIR'] = '/tmp'
os.environ['MPLCONFIGDIR'] = '/tmp'
os.environ['NUMBA_NUM_THREADS'] = '1' # Evitar conflictos de CPU

import numpy as np
import librosa
import io
import soundfile as sf
import requests
import tensorflow as tf
import gc # Importante para limpiar RAM
import time
from flask import Flask, request, jsonify

# --- CONFIGURACIÓN ---
MODEL_FILE = 'modelo_ratas.tflite'
UBIDOTS_TOKEN = "BBUS-05HpL3CGv101KvETp3hGXsPHSGQuJ6"
DEVICE_LABEL = "bugbeats"
VARIABLE_LABEL = "rata"
MIN_CONFIDENCE = 0.80 

app = Flask(__name__)
interpreter = None
input_details = None
output_details = None

print("--- 🤖 SERVIDOR NEURONAL (TF CPU + FIX NUMBA) ---")

# Cargar modelo UNA sola vez al inicio
try:
    if os.path.exists(MODEL_FILE):
        interpreter = tf.lite.Interpreter(model_path=MODEL_FILE)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ Cerebro cargado.")
    else:
        print(f"❌ ERROR: Falta {MODEL_FILE}")
except Exception as e:
    print(f"❌ ERROR CARGA: {e}")

def procesar_audio(audio_data):
    # Normalización
    max_val = np.max(np.abs(audio_data))
    if max_val > 0: audio_data = audio_data / max_val

    # MFCCs
    # print("   (Calculando MFCCs...)") 
    mfccs = librosa.feature.mfcc(y=audio_data, sr=16000, n_mfcc=40)
    features = np.mean(mfccs.T, axis=0)
    return np.array([features], dtype=np.float32)

@app.route('/', methods=['GET'])
def home():
    return "BugBeats AI Alive"

@app.route('/detectar', methods=['POST'])
def detectar():
    global interpreter
    t_inicio = time.time()
    
    if interpreter is None: return jsonify({"error": "Modelo off"}), 500

    print(f"\n📞 Recibido: {len(request.data)} bytes")
    
    try:
        # 1. Leer WAV
        print("-> 1. Decodificando audio...")
        data, samplerate = sf.read(io.BytesIO(request.data))
        
        # Si es stereo, pasar a mono
        if len(data.shape) > 1: data = data.mean(axis=1)
        if data.dtype != 'float32': data = data.astype('float32')

        # 2. Inferencia
        print("-> 2. Extrayendo características (Librosa)...")
        input_data = procesar_audio(data)
        
        print("-> 3. Ejecutando Red Neuronal...")
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prob_rata = float(output_data[0][0])
        
        t_total = time.time() - t_inicio
        print(f"✅ FINALIZADO en {t_total:.2f}s | Confianza: {prob_rata*100:.1f}%")

        # 3. Ubidots (Con timeout corto para no bloquear)
        if prob_rata >= MIN_CONFIDENCE:
            try:
                print("-> 4. Enviando a Ubidots...")
                requests.post(
                    f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}",
                    json={VARIABLE_LABEL: 1.0},
                    headers={"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"},
                    timeout=2
                )
            except: pass
        else:
             try:
                requests.post(
                    f"https://stem.ubidots.com/api/v1.6/devices/{DEVICE_LABEL}",
                    json={VARIABLE_LABEL: 0.0},
                    headers={"X-Auth-Token": UBIDOTS_TOKEN, "Content-Type": "application/json"},
                    timeout=2
                )
             except: pass

        # 4. LIMPIEZA DE MEMORIA OBLIGATORIA
        del data
        del input_data
        gc.collect() 

        return jsonify({
            "status": "ok", 
            "es_rata": 1 if prob_rata >= MIN_CONFIDENCE else 0,
            "mensaje": "RATA 🐀" if prob_rata >= MIN_CONFIDENCE else "AMBIENTE 🍃",
            "confianza": prob_rata
        })

    except Exception as e:
        print(f"❌ Error en proceso: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
