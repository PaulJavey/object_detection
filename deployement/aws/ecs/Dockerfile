# Image de base avec CUDA pour le support GPU
FROM nvidia/cuda:11.7.1-cudnn8-runtime-ubuntu20.04 AS base

# Variables d'environnement
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    TZ=UTC

# Installer les dépendances système
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.9 \
    python3-pip \
    python3.9-dev \
    python3-setuptools \
    python3-wheel \
    libgl1-mesa-glx \
    libglib2.0-0 \
    git \
    && rm -rf /var/lib/apt/lists/*

# Définir le répertoire de travail
WORKDIR /app

# Copier les fichiers requis
COPY requirements.txt .

# Installer les dépendances Python
RUN pip3 install --no-cache-dir -r requirements.txt

# Copier le code source
COPY . .

# Télécharger les modèles pré-entraînés
RUN mkdir -p data/models && \
    python -c "import torch; from pathlib import Path; torch.hub.download_url_to_file('https://github.com/ultralytics/yolov5/releases/download/v6.1/yolov5s.pt', Path('data/models') / 'yolov5s.pt')"

# Exposer les ports pour l'API et l'interface web
EXPOSE 8000 8050

# Image finale optimisée
FROM base AS runtime

# Arguments de construction
ARG BUILD_DATE
ARG VERSION
ARG VCS_REF

# Labels pour les métadonnées
LABEL org.label-schema.build-date=${BUILD_DATE} \
      org.label-schema.name="object-detection-yolov5" \
      org.label-schema.description="Système de surveillance utilisant YOLOv5 pour la détection d'objets en temps réel" \
      org.label-schema.vcs-ref=${VCS_REF} \
      org.label-schema.vcs-url="https://github.com/username/object_detection_yolov5" \
      org.label-schema.version=${VERSION} \
      org.label-schema.schema-version="1.0" \
      maintainer="Your Name "

# Commande par défaut
CMD ["python", "-m", "src.api.server"]
