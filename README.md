# Sign2Notes — Local Demo

## Prerequisites
- Python 3.10+
- Node.js 18+
- NVIDIA drivers + CUDA for RTX 4050 (recommended)
- GPU memory is limited; use small batch sizes and seq_len during training.

## Setup (shell)
```bash
# from project root
python -m venv .venv
source .venv/bin/activate

# install python deps
pip install -r ml_service/requirements.txt

# install backend deps
cd backend
npm install
cd -

# frontend deps
cd frontend/react-app
npm install
cd -
