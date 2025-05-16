FROM python:3.12

WORKDIR /app

# Installer les dépendances système
RUN apt-get update && apt-get install -y \
    git \
    build-essential \
    cmake \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && apt-get clean

# Mettre à jour pip et installer wheel
RUN pip install --upgrade pip wheel

# Installer PyTorch + CUDA 11.8 (versions selon ton fichier requirements)
RUN pip install torch==2.7.0+cu118 torchvision==0.22.0+cu118 torchaudio==2.7.0+cu118 --index-url https://download.pytorch.org/whl/cu118

# Cloner detectron2 et installer en mode editable
RUN git clone https://github.com/facebookresearch/detectron2.git && \
    pip install -e detectron2

# Copier requirements-base.txt et installer le reste
COPY requirements-base.txt .
RUN pip install --no-cache-dir -r requirements-base.txt

# Copier le code source
COPY ./app ./app

EXPOSE 8001

CMD ["uvicorn", "app.api.main:app", "--host", "0.0.0.0", "--port", "8001", "--reload", "--reload-dir", "app"]
