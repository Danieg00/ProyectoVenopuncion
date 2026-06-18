# Sistema de Entrenamiento en Venopunción AR

Sistema de detección en tiempo real para identificar el brazo y el tablero ChArUco, con captura desde Raspberry Pi 3B+ (Camera Module 3 NoIR) y visualización en Unity para entrenamiento en venopunción con realidad aumentada.

## Quick Start

### 1. Instalar dependencias (lado PC)

```bash
git clone https://github.com/Danieg00/ProyectoVenopuncion.git
cd ProyectoVenopuncion
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Preparar la Raspberry Pi

Sistema operativo recomendado: **Raspberry Pi OS Lite (64-bit) Legacy (Bookworm)**, flasheado en un pendrive de 16 GB (la Pi 3B+ no tiene almacenamiento interno). Se eligió la versión Lite para liberar RAM y CPU al eliminar el entorno gráfico, dejando todos los recursos para capturar y codificar vídeo. Bookworm es necesario porque introduce `libcamera`/`rpicam-apps`; las versiones basadas en Bullseye usan el sistema obsoleto `raspistill`.

Conecta el cable de la cámara al puerto CSI con los contactos plateados orientados hacia el puerto HDMI, y alimenta la Pi con al menos **5V / 2.5A** (se recomienda una batería portátil para mantener un suministro estable y permitir movilidad durante las pruebas).

Verifica la conexión por SSH y la detección de la cámara:

```bash
ssh usuario@picam.local
rpicam-hello --list-cameras   # debe listar el sensor imx708_noir
rpicam-still -o test_image.jpg
```

Para apagar de forma segura (el sistema corre desde un pendrive):

```bash
sudo shutdown -h now
```

### 3. Generar el tablero ChArUco

```bash
python charuco_board_generator.py
```

Esto genera `charuco_board.png`. Imprimir a **escala 100%** en cartulina satinada y montar sobre una superficie rígida a ser posible.

### 4. Ejecutar el pipeline

**Lado Raspberry Pi** — transmite vídeo H.264 por UDP:

```bash
rpicam-vid -t 0 --inline --width 1536 --height 864 --framerate 30 --codec h264 -o udp://<PC_IP>:5000
```

**Lado PC** — recibe el stream, procesa y envía a Unity:

```bash
./venv/bin/python trial/pc_udp_video_bridge.py --unity-host <UNITY_IP>
```

También existe un camino alternativo basado en HTTP (`trial/pc_inference_api.py`) que acepta fotogramas JPEG sueltos; se mantiene como opción de prueba separada, pero el flujo principal de trabajo es el UDP descrito arriba.

## Estructura del Proyecto

```
├── trial/                   # Scripts de transporte e inferencia (tests extremo a extremo)
│   ├── pc_udp_video_bridge.py   # Bridge principal: recibe UDP, procesa, envía a Unity
│   └── pc_inference_api.py      # API HTTP alternativa (JPEG -> JSON Unity)
│
├── models/                  # Lógica de detección
│   ├── charuco_detector.py # Detección de tablero ChArUco + pose 3D
│   ├── arm_segmenter.py    # Segmentación de brazo (depth / MediaPipe / HSV)
│   └── vein_detector.py    # Detector de venas (placeholder, Fase 2)
│
├── utils/                   # Utilidades
│   ├── data_export.py      # Exportación JSON/CSV por fotograma
│   └── unity_export.py     # Empaquetado del envelope JSON + envío UDP a Unity
│
├── unity/                   # Integración Unity
│   ├── DetectionReceiver.cs        # Receptor UDP en puerto 5000
│   └── VenopunctureVisualization.cs # Parseo y visualización (tablero, brazo, venas)
│
├── calibration/             # Matrices de calibración de cámara (auto-pobladas)
├── output/                  # Fotogramas, vídeos y datos JSON/CSV generados
├── demo.py                  # Genera una escena sintética sin cámara conectada
├── main.py                  # Pipeline principal
├── charuco_board_generator.py
└── TRIAL_API_RUNBOOK.md     # Documentación del esquema vein (campo opcional)
```

## Estado de los componentes

| Componente | Estado |
|---|---|
| Detección de tablero ChArUco y pose 3D | ✅ Implementado |
| Segmentación de brazo (MediaPipe + fallback HSV) | ✅ Implementado |
| Exportación de datos (JSON/CSV) | ✅ Implementado |
| Transporte Pi → PC → Unity (UDP) | ✅ Implementado |
| Receptor y visualización en Unity | ✅ Implementado |
| Demo sin cámara | ✅ Implementado |
| Detección real de venas | 🔄 Infraestructura lista, algoritmo pendiente |
| Iluminación NIR dedicada (LEDs 940nm + filtro) | 🔄 Pendiente |
| Calibración de cámara para la Pi | 🔄 Pendiente |
| Configuración estereoscópica (3D AR) | 🔄 En evaluación |

## API Rápida

### Detectar tablero

```python
from models.charuco_detector import CharucoDetector

detector = CharucoDetector()
detection, pose = detector.detect_and_estimate_pose(frame)

if detection.success:
    print(f"Tablero detectado a distancia: {pose.distance}m")
```

### Segmentar brazo

```python
from models.arm_segmenter import ArmSegmenter

segmenter = ArmSegmenter(use_depth=False, use_ml=True)
arm_result = segmenter.segment_arm(frame, depth_frame=None)

if arm_result.success:
    print(f"Bounding box del brazo: {arm_result.bbox}")
```

### Exportar datos

```python
from utils.data_export import DetectionDataExporter

exporter = DetectionDataExporter()
exporter.export_frame(detection, pose, arm_result)
exporter.close()
```

## Hardware utilizado

- **Cámara**: Raspberry Pi Camera Module 3 NoIR (sensor `imx708_noir`)
- **Placa**: Raspberry Pi 3B+
- **Almacenamiento**: Pendrive USB 16GB (Raspberry Pi OS Lite 64-bit, Bookworm)
- **Alimentación**: Batería portátil USB, mínimo 5V / 2.5A

## Licencias y atribuciones

Este proyecto usa:
- **OpenCV** (Apache 2.0)
- **MediaPipe** (Apache 2.0)
- **NumPy** (BSD)

Arquitectura de referencia para la detección de venas: [CUBITAL](https://github.com/EdwinTSalcedo/CUBITAL) (U-Net entrenada con TensorFlow).

---

# Informe del Proyecto

## 1. Situación Actual

En este punto el código para la detección del tablero ChArUco está implementado y el brazo puede ser localizado sobre él. El siguiente problema a resolver es la detección de venas para comenzar el modelado 3D del escenario completo, pero antes de llegar a eso ha sido necesario establecer toda la cadena de captura y transporte de vídeo desde la Raspberry Pi hasta Unity. Además, las imágenes deben ser procesadas por algún sistema de Machine Learning.

La detección se basa en las propiedades ópticas de la hemoglobina, que absorbe la luz NIR, por lo que es conveniente usar algún tipo de fuente de luz infrarroja dedicada. El camino seguido hasta ahora está centrado en la Raspberry Pi 3B+ con la cámara Module 3 NoIR y no depende todavía de lógica de visualización AR.

## 2. Montaje Físico y Puesta en Marcha

Para poder trabajar con la cámara fue necesario pasar por varias fases antes de tener el sistema funcionando de forma estable.

**Conexión de la cámara y alimentación.** El primer paso fue insertar el cable en el puerto CSI de la Pi, con los contactos plateados orientados hacia el puerto HDMI. La pestaña es algo rígida y mostraba resistencia al inicio, pero tras varios intentos se pudo asegurar totalmente. En cuanto a la alimentación, en el material no estaba incluido el cable micro-USB ni el transformador para dar corriente al dispositivo; es necesario que reciba al menos 5V y 2.5A. Para ello busqué un cable compatible y me decanté por una batería portátil como fuente de alimentación, al poder mantener un soporte constante de electricidad y la movilidad necesaria para fotos o vídeos de prueba.

**Sistema operativo.** Las Raspberry no cuentan con almacenamiento interno en la placa, así que es necesario usar una tarjeta de memoria o, en mi caso, un pendrive. Le hice flash con el sistema operativo "Raspberry Pi OS Lite (64-bit) Legacy (Bookworm)" en un pen de 16 GB. La razón de elegir la versión Lite es la ligereza al eliminar el entorno gráfico de escritorio y liberar una cantidad considerable de la limitada RAM de 1 GB de la Pi 3B+, dejando todos los recursos dedicados exclusivamente a capturar y codificar vídeo. Con el escritorio corriendo en segundo plano estaría más limitado. Se añade además el problema de las librerías de uso de la cámara, para lo que se requería esa versión Bookworm al leer que podrían haber problemas en otras versiones: en el caso de Bullseye usaban un sistema obsoleto (`raspistill`), mientras que Bookworm introduce el nuevo estándar (`libcamera` y `rpicam-apps`).

**Conexión SSH.** Al usar `ssh usuario@picam.local` se tiene acceso a la Pi. La siguiente tabla explica paso a paso cómo se prepara:

| # | Paso | Qué ocurre |
|---|---|---|
| 1 | Pre-configuración | Al flashear la imagen con Raspberry Pi Imager se preescribieron el nombre de la red Wi-Fi, la contraseña y el nombre de usuario, y se activó SSH. |
| 2 | Conexión | Al arrancar, la Pi se conectó a la red Wi-Fi e inició el servidor OpenSSH en segundo plano automáticamente. Al ejecutar `ping picam.local` en el equipo se pudo recibir respuesta. |
| 3 | Primeros pasos | Una vez introducida la contraseña, la Pi abrió la pasarela y apareció el prompt para empezar a operar dentro de ella. Llegado este punto lo recomendable es ejecutar `sudo apt update && sudo apt upgrade -y`. |

**Validación del hardware y apagado seguro.** Una vez dentro vía SSH, lo primero fue confirmar que el sistema reconocía la cámara con `rpicam-hello --list-cameras`, que devolvió el sensor `imx708_noir` correctamente detectado. Después capturé una fotografía de prueba con `rpicam-still -o test_image.jpg` para validar el funcionamiento. Es posible traer esa imagen al equipo de forma segura con `scp username@picam.local:~/test_image.jpg ~/Desktop/` (en Linux) o su equivalente en Windows. Dado que el sistema operativo corre desde un pendrive y la sesión SSH está interactuando activamente con él, no se puede simplemente desconectar el cable: para evitar corrupción del sistema de ficheros conviene apagar siempre con `sudo shutdown -h now`.

## 3. Pipeline

El flujo completo, una vez montado y funcionando, opera de la siguiente manera:

1. La Raspberry Pi captura vídeo desde la cámara NoIR.
2. `rpicam-vid` codifica y transmite el vídeo en H.264 al PC vía UDP.
3. El bridge en el PC recibe el stream y decodifica los fotogramas con OpenCV.
4. Un paso de análisis ligero extrae una bounding box aproximada del brazo.
5. El bridge empaqueta los resultados en JSON con el formato que espera Unity (array `detections`).
6. El JSON se envía a Unity vía UDP por el puerto 5000.
7. Unity recibe el JSON y actualiza la escena.

## 4. Implementado y Listo

**Detección del tablero ChArUco y estimación de pose.** El detector en `models/charuco_detector.py` es el componente más completo del sistema. Detecta el tablero 7×5 con una precisión de 3–5 mm y estima la pose 3D completa (vectores de rotación y traslación) usando la matriz de calibración y los coeficientes de distorsión. Durante la ejecución se visualizan los ejes de coordenadas en tiempo real (X=rojo, Y=verde, Z=azul). Está integrado tanto en `main.py` como en el pipeline de demostración.

**Segmentación del brazo.** El segmentador en `models/arm_segmenter.py` implementa tres métodos y selecciona automáticamente el más potente disponible: segmentación basada en profundidad (pensada para cámaras RGB-D como RealSense o Kinect; con la Pi Camera Module 3 NoIR este método no aplica, ya que se trata de una cámara monocular sin sensor de profundidad por hardware), MediaPipe Holistic como método principal en el setup actual, y detección de piel por espacio de color HSV como fallback. La salida en todos los casos es la misma: máscara binaria, contornos y bounding box del brazo, lo que permite que el resto del pipeline funcione sin importar qué método se usa.

**Exportación de datos y comunicación con Unity.** `utils/data_export.py` escribe las detecciones por fotograma en JSON y CSV. `utils/unity_export.py` las empaqueta en el envelope que espera Unity y las envía por UDP. El campo `vein` ya está presente en todo el payload desde el principio, lo que significa que cuando se implemente la detección real de venas no habrá que tocar el esquema ni el código de Unity.

**Receptor y visualización en Unity.** `unity/DetectionReceiver.cs` escucha en el puerto UDP 5000 y `unity/VenopunctureVisualization.cs` analiza el payload y actualiza la escena: tablero, bounding box del brazo y, cuando llegue el momento, el contorno de las venas. El path de renderizado de las mismas ya está definido en el código, simplemente se activa cuando hay datos reales disponibles. En la escena el tablero es perfectamente visible; el brazo por ahora usa un placeholder, suficiente para hacer pruebas.

**Demostración sin cámara.** Con `demo.py` se puede generar una escena sintética con tablero y brazo simulados para probar la detección y la exportación sin necesidad de tener la cámara conectada. Esto fue útil antes de contar con el material físico, para validar el pipeline de datos de forma independiente al hardware.

## 5. Procesamiento de Imagen

En cuanto al procesamiento de la imagen NIR, la cadena que se plantea una vez resuelto el hardware es la siguiente:

1. Conversión a escala de grises y aplicación de CLAHE (Contrast Limited Adaptive Histogram Equalization) para mejorar la visibilidad de las venas.
2. Uso de modelos de Deep Learning basados en redes neuronales convolucionales (CNN), implementados con TensorFlow. El propio repositorio de referencia ([CUBITAL](https://github.com/EdwinTSalcedo/CUBITAL)) usa una arquitectura U-Net entrenada con este framework para segmentar las venas a nivel de píxel, y según la literatura consultada ofrece resultados de precisión del 83%. En ese mismo repositorio se probaron además SegNet, PSPNet, Pix2Pix y DeepLabv3+, siendo U-Net la que obtuvo mejor precisión, lo que llevó a centrar el resto de la investigación en ese método para detectar la fosa antecubital. Cabe aclarar que para esta fase no es conveniente usar herramientas de orquestación como n8n: están pensadas para conectar APIs y mover datos entre servicios, no para ejecutar modelos de inferencia. La segmentación trabaja píxel a píxel y requiere un motor de deep learning real, no un flujo de automatización; la recomendación para obtener buenos resultados es seguir con TensorFlow.
3. Integración mediante OpenCV para el tratamiento de imagen, que se puede usar con Unity a través de la suscripción.

Con una sola cámara es suficiente para detección 2D y proyección sobre pantalla, y es más simple de calibrar y consume menos recursos de procesamiento. Sin embargo, dado que la idea es simular un entorno 3D en AR, se debería valorar una configuración estereoscópica con dos cámaras. Aumenta la complejidad del software porque requiere calibración estéreo, pero daría mucha más información de profundidad y a la hora de simular tendría más fiabilidad para la imagen.

## 6. Estado de la Detección de Venas

El proyecto ya contiene toda la infraestructura software para las venas, pero el algoritmo de detección en sí todavía no está implementado. Lo que existe actualmente es: `models/vein_detector.py` con la clase del detector y la API prevista; `unity/VenopunctureVisualization.cs` con `VeinData`, un line renderer y el path `ShowVeins` para dibujar contornos; los scripts de `trial/pc_inference_api.py` y `trial/pc_udp_video_bridge.py`, que ya emiten el objeto `vein` en el JSON.

Lo que falta implementar es la extracción real de venas a partir de los datos de imagen NIR, la puntuación de confianza desde un modelo real, y el mapeo 3D de las venas si se requiere salida tridimensional. El camino software está prácticamente completo, pero la detección en sí es la siguiente fase, que necesita aclarar tanto la tecnología hardware (en caso de querer visualizar en AR) como el software para el motor de ML.

De cara a avanzar, la precisión debe ser inferior a 5 mm para ser clínicamente útil, por lo que el hardware y el algoritmo tienen que estar a la altura de ese requisito. Además es fundamental controlar la iluminación: una matriz de LEDs de 940 nm para iluminar el área y bloquear el resto del espectro visible en la mayor medida posible. Posteriormente se puede empezar a modelar en Unity sabiendo con qué información contamos.

## 7. Próximos Pasos

Para contar con algo funcional según la estructura de lo que se tiene ahora mismo y el material hardware al alcance, los pasos son:

1. Verificar que el stream de la Pi llega al PC de forma fiable en la red local.
2. Confirmar que Unity recibe el JSON sin errores de deserialización.
3. Registrar una ejecución de prueba corta y conservar los ficheros JSON de salida como referencia.
4. Añadir calibración de cámara solo después de que la ruta de transporte sea estable.
5. Sustituir el placeholder del detector de venas con un algoritmo real basado en CNN e iluminación NIR.
6. Incorporar la iluminación infrarroja (LEDs 940 nm + filtro de luz) al setup físico.
7. Valorar la configuración estereoscópica para el entorno 3D AR una vez consolidado el pipeline con una sola cámara.
