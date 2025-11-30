import os
import numpy as np
import librosa
import io
import wave
import soundfile as sf
import requests
import tflite_runtime.interpreter as tflite 
from flask import Flask, request, jsonify

# --- CONFIGURACIÓN ---
MODEL_FILE = 'modelo_ratas.tflite'
UBIDOTS_TOKEN = "BBUS-05HpL3CGv101KvETp3hGXsPHSGQuJ6"
DEVICE_LABEL = "bugbeats"
VARIABLE_LABEL = "rata"

# Las redes neuronales son más seguras. 
# Si tu prueba local dio muy buenos resultados, podemos confiar en 80%
MIN_CONFIDENCE = 0.80 

app = Flask(__name__)
interpreter = None
input_details = None
output_details = None

print("--- 🤖 SERVIDOR NEURONAL (TFLITE) ---")

# CARGAR MODELO
try:
    if os.path.exists(MODEL_FILE):
        interpreter = tflite.Interpreter(model_path=MODEL_FILE)
        interpreter.allocate_tensors()
        
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        print(f"✅ Cerebro cargado: {MODEL_FILE}")
    else:
        print(f"❌ ERROR: No encuentro '{MODEL_FILE}'")
except Exception as e:
    print(f"❌ ERROR CRÍTICO AL CARGAR TFLITE: {e}")

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
    return "BugBeats Neural Server Active 🧠"

@app.route('/detectar', methods=['POST'])
def detectar():
    if interpreter is None: return jsonify({"error": "Cerebro desconectado"}), 500

    print(f"\n📞 Audio recibido: {len(request.data)} bytes")
    
    # 1. LEER EL AUDIO (WAV)
    try:
        data, samplerate = sf.read(io.BytesIO(request.data))
    except:
        # Fallback para RAW si fuera necesario
        try:
            wav_buffer = io.BytesIO()
            with wave.open(wav_buffer, "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(request.data)
            wav_buffer.seek(0)
            data, samplerate = sf.read(wav_buffer)
        except: return jsonify({"error": "Formato inválido"}), 400

    try:
        if data.dtype != 'float32': data = data.astype('float32')

        # 2. PRE-PROCESAMIENTO (Igual que en el entrenamiento)
        # Normalizamos volumen al máximo (para que la IA escuche bien)
        max_val = np.max(np.abs(data))
        if max_val > 0:
            data = data / max_val

        # Extraer MFCC (40 características)
        mfccs = librosa.feature.mfcc(y=data, sr=16000, n_mfcc=40)
        features = np.mean(mfccs.T, axis=0)
        
        # Preparar tensor para TFLite (Shape: [1, 40])
        input_data = np.array([features], dtype=np.float32)
        
        # 3. EJECUTAR INFERENCIA
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        
        # 4. OBTENER RESULTADO
        output_data = interpreter.get_tensor(output_details[0]['index'])
        prob_rata = float(output_data[0][0]) # Probabilidad de 0.0 a 1.0
        
        print(f"🧠 Confianza Neuronal: {prob_rata*100:.2f}% Rata")

        es_rata = (prob_rata >= MIN_CONFIDENCE)
        
        enviar_a_ubidots(es_rata)
        
        return jsonify({
            "status": "ok", 
            "es_rata": 1 if es_rata else 0,
            "mensaje": "RATA 🐀" if es_rata else "AMBIENTE 🍃",
            "confianza": prob_rata
        })

    except Exception as e:
        print(f"❌ Error procesando: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 10000))
    app.run(host='0.0.0.0', port=port)
