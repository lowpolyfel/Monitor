#!/usr/bin/env bash
set -euo pipefail
# Ejecuta la interfaz gráfica (Tkinter). Requiere entorno con display (X11).
PYTHONPATH=src python3 -c "from rdm_monitor.gui import main; main()"
