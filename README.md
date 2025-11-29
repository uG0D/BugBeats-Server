# 🐭Sistema IoT de Detección Acústica para Alerta Temprana de Plagas (BugBeats)
## 💻Curso: Arquitectura del computador
## 👩‍💻Integrantes:
- Hugo Barboza (Programador) -> 25101714
- Daniela Ricapa (Encargada de la conexión con la nube) -> 25100739
- Cielo Valle (Documentadora) -> 25102283
## 📆Fecha 
Noviembre 2025
## ✒ 1. Resumen del proyecto
El sistema ADPS-Lite permite detectar ruidos anómalos producidos por plagas (como roedores) en ambientes de almacenamiento. Utiliza un micrófono I2S conectado a una Raspberry Pi Pico W que calcula la energía acústica en tiempo real. Si el nivel de ruido supera un umbral, activa una alerta visual (LED) y envía un aviso a la nube mediante ThingSpeak.
## 💿 2. Arquitectura del sistema (⭕)
*Subir diagrama de blques*
Flujo general:
1. El micrófono I2S capta señales acústicas.
2. El microcontrolador calcula el valor RMS.
3. Si el nivel supera el umbral, se activa el LED y se envía una alerta al servidor IoT.
4. ThingSpeak registra y grafica los eventos.
5. Herramienta sugerida: Lucidchart o Draw.io
## 🧠 3. Componentes Utilizados 
| Componente | Descripción | Imagen |
|-------------|-------------|--------|
| Raspberry Pi Pico W | Microcontrolador principal con WiFi integrado. | <img src="img/raspberrypi.jpg" width="180"> |
| Micrófono I2S INMP441 | Sensor de audio digital con salida I2S. | <img src="img/s-l400.png" width="180"> |
| LED + resistencia 220Ω | Alerta visual de detección acústica. | <img src="img/Led.Verde_.webp" width="180"> |

## 💻 4. Código Fuente
📂 Ubicación: src/main.py
# Lectura de micrófono I2S y detección de ruidos
import machine, math, time

*Parte del código*

🗒️ El código completo está disponible en el repositorio con comentarios detallados.

## 🧩 5. Diagrama de Flujo (⭕)
*Aquí debo pegar la imagen de mi diagrama de flujo*
*🔹 Debes subir la imagen (por ejemplo flujo.png) a una carpeta dentro de tu repositorio — normalmente /docs/ o /assets/.*
*![Diagrama de flujo del sistema](./assets/flujo.png)*

Descripción del proceso:
- Inicio
- Lectura de señal I2S
- Cálculo RMS
- Comparar con umbral
- Si RMS > umbral → activar LED + enviar alerta
- Si RMS ≤ umbral → continuar monitoreo

## 🔌 6. Diagrama de Conexiones (Fritzing) (⭕)
*Subir imagen del diagrama en Fritzing*
| Elemento      | Pin Pico         | Descripción               |
| ------------- | ---------------- | ------------------------- |
| Micrófono I2S | GP10, GP11, GP12 | Datos, reloj, LRCLK       |
| LED           | GP15             | Alerta visual             |
| GND/VCC       | –                | Alimentación y referencia |

## ☁️ 7. Conectividad IoT
- Plataforma: ThingSpeak
- Protocolo: HTTP GET
- Frecuencia de envío: cada vez que RMS > umbral
- Campos registrados: Nivel de sonido (RMS), Timestamp
## 🧱 8. Diseño 3D del Case (⭕)
*Imagen del diseño 3D*
- Software: Tinkercad
- Diseño modular con ventilación lateral
- Orificios para LED y puerto micro-USB
📁 Archivo .STL disponible en /3D/ADPS_Lite_case.stl
## 🎥 9. Video Demostrativo (⭕)
*🔗 Ver video en YouTube (Aquí debe ir el link del video colgado en YouTube)*
Duración: 5:00 min
Contenido: presentación, prototipo en acción, explicación técnica, conclusiones
## 📊 10. Póster Técnico
📁 Archivo digital: /poster/ADPS_Lite_Poster.pdf
📐 Formato A2 – incluye metodología, resultados y arquitectura.
## 🗂️ 11. Gestión de Proyecto (⭕)
*documento 📄 Google Sheets de Tareas*
Incluye responsable, avance, estado y fechas.
## 🧾 12. Conclusiones
- Se logró implementar un sistema funcional de detección acústica en tiempo real.
- El diseño es escalable y adaptable a entornos agrícolas o urbanos.
- Futuras mejoras: agregar modelo de ML para clasificación de ruidos, alimentación por batería recargable y módulo de notificaciones vía Telegram.
## 🔗 13. Referencias
- Raspberry Pi Pico W Documentation
- ThingSpeak API Docs
- INMP441 Datasheet
