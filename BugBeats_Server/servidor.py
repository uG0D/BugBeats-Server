from flask import Flask, request, jsonify
import sys

app = Flask(__name__)

print("--- ⚡ SERVIDOR DIAGNÓSTICO INICIADO ⚡ ---")

@app.route('/', methods=['GET'])
def home():
    return "Diagnostico Online"

@app.route('/detectar', methods=['POST'])
def detectar():
    # Imprimir en los logs de Render para confirmar llegada
    print(f"📞 Petición recibida desde: {request.remote_addr}")
    print(f"📦 Tamaño de datos: {len(request.data)} bytes")
    
    # Responder inmediatamente
    return jsonify({
        "status": "ok",
        "mensaje": "CONEXIÓN EXITOSA",
        "es_rata": 0,
        "confianza": 1.0
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=10000)
