# Taller RL - Reinforcement Learning aplicado a Hardware

## Setup

Necesitas Python 3.11 o 3.12 instalado en tu sistema.

### 1. Instalar uv

```bash
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"

# Mac / Linux
curl -LsSf https://astral.sh/uv/install.sh | sh

brew install uv # MacOS con Homebrew

sudo pacman -S uv # Arch Linux
```

Cierra y vuelve a abrir la terminal tras la instalacion.

### 2. Instalar dependencias

```bash
cd gym-rl
uv sync
```

### 3. Abrir el notebook

```bash
# Opcion A: Jupyter
uv run jupyter notebook taller_rl.ipynb

# Opcion B: VS Code
# Abre taller_rl.ipynb y selecciona el kernel .venv/Scripts/python
```
