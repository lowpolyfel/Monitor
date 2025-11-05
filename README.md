# Monitor

Herramienta para detectar actividad en video utilizando diferentes motores de movimiento
(OpenCV). El proyecto incluye una interfaz de línea de comandos, una interfaz gráfica
básica y utilidades para registrar intervalos de operación.

## Características principales

- Motores de detección configurables (MOG2, diferencias de cuadros, KNN, flujo óptico,
  promedios móviles y bordes).
- Exportación opcional de video anotado y registro de intervalos en CSV.
- Interfaz gráfica en Tkinter para seleccionar la fuente de video y parámetros comunes.
- Scripts de ejecución listos para producción.

## Requisitos

- Python 3.10 o superior.
- Dependencias listadas en `requirements.txt`:
  - `opencv-python`
  - `numpy`

Instala las dependencias con:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
pip install -e .
```

> El paso `pip install -e .` registra el paquete `src` usando el layout "src only" y
> evita depender de variables de entorno como `PYTHONPATH`.

## Uso rápido

### Línea de comandos

Una vez instalado el paquete, ejecuta el monitor desde cualquier ubicación:

```bash
python -m src --src data/raw_videos/MI_VIDEO.mp4 --idle_stop_sec 20 \
  --out outputs/annotated/MI_VIDEO_annotated.mp4
```

Para ver todas las opciones disponibles:

```bash
python -m src --help
```

Si ejecutas la herramienta sin `--src`, aparecerá un menú interactivo que permite elegir una cámara o seleccionar un directorio específico para explorar videos.

Si prefieres no instalar el paquete, puedes ejecutar la CLI directamente con:

```bash
python main.py
```

### Interfaz gráfica

Ejecuta la GUI (requiere un entorno con servidor gráfico disponible):

```bash
./scripts/run_gui.sh
```

La interfaz permite escoger el directorio de videos antes de iniciar el análisis.

### Script continuo

Procesa un video completo con parámetros predefinidos:

```bash
./scripts/run_continuo.sh data/raw_videos/VIDEO.mp4
```

## Estructura del repositorio

```
configs/        # Configuraciones de ejemplo (YAML)
legacy/         # Scripts de prueba históricos
scripts/        # Scripts de conveniencia para CLI/GUI
src/            # Código fuente del paquete (estructura plana)
```

## Pruebas heredadas

Los scripts `legacy/testv1.py` y `legacy/testv2.py` se conservan sólo como referencia
histórica y no forman parte del flujo de pruebas actual.

## Desarrollo

1. Crea un entorno virtual y sigue los pasos de instalación indicados arriba.
2. Ejecuta `python main.py` o `python -m src` para validar la CLI.
3. Utiliza `python -m src.gui` para probar la interfaz gráfica.

Se aceptan contribuciones mediante pull requests. Asegúrate de ejecutar herramientas
estáticas o pruebas relevantes antes de enviar cambios.
