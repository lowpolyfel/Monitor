# rdm-monitor (Raspberry Pi + OpenCV)

## Ejecución rápida (sin instalar el paquete)
Desde la raíz del proyecto (`~/rdm`):

```bash
PYTHONPATH=src python3 -m rdm_monitor --src data/raw_videos/MI_VIDEO.mp4 --idle_stop_sec 20 --out outputs/annotated/MI_VIDEO_annotated.mp4


abrir menu simple
PYTHONPATH=src python3 -m rdm_monitor


ENTRAR EN MODO UI
./scripts/run_gui.sh

