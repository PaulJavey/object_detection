#!/bin/bash
set -e

# Configuration
MODELS_DIR="data/models"
MODELS=(
    "yolov5n"    # Plus petit et plus rapide
    "yolov5s"    # Petit
    "yolov5m"    # Moyen
    "yolov5l"    # Grand
    "yolov5x"    # Extra grand
)
DEFAULT_MODEL="yolov5s"

# Couleurs pour les sorties
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Fonction pour afficher les étapes
log() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

# Fonction pour afficher les avertissements
warn() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

# Fonction pour afficher les erreurs
error() {
    echo -e "${RED}[ERROR]${NC} $1"
    exit 1
}

# Créer le répertoire des modèles s'il n'existe pas
create_models_directory() {
    if [ ! -d "$MODELS_DIR" ]; then
        log "Création du répertoire $MODELS_DIR"
        mkdir -p "$MODELS_DIR"
    fi
}

# Télécharger un modèle spécifique
download_model() {
    local model=$1
    local model_file="${MODELS_DIR}/${model}.pt"

    if [ -f "$model_file" ]; then
        log "Le modèle $model existe déjà."
    else
        log "Téléchargement du modèle $model..."

        # Utiliser Python pour télécharger le modèle via torch.hub
        python3 -c "
import torch
import os
from pathlib import Path

torch.hub.download_url_to_file(
    f'https://github.com/ultralytics/yolov5/releases/download/v6.1/${model}.pt',
    Path('${model_file}')
)

if os.path.exists('${model_file}'):
    print(f'${GREEN}Modèle ${model} téléchargé avec succès.${NC}')
else:
    print(f'${RED}Échec du téléchargement du modèle ${model}.${NC}')
    exit(1)
"
    fi
}

# Afficher l'aide
show_help() {
    echo "Script pour télécharger les modèles YOLOv5"
    echo
    echo "Usage: $0 [options]"
    echo
    echo "Options:"
    echo "  -h, --help         Afficher cette aide"
    echo "  -a, --all          Télécharger tous les modèles"
    echo "  -m, --model MODEL  Télécharger un modèle spécifique (yolov5n, yolov5s, yolov5m, yolov5l, yolov5x)"
    echo
    echo "Par défaut, seul le modèle $DEFAULT_MODEL est téléchargé si aucune option n'est spécifiée."
    echo
}

# Vérifier les dépendances
check_dependencies() {
    log "Vérification des dépendances..."

    if ! command -v python3 &> /dev/null; then
        error "Python 3 n'est pas installé. Veuillez l'installer."
    fi

    # Vérifier si torch est installé
    if ! python3 -c "import torch" &> /dev/null; then
        warn "PyTorch n'est pas installé. Installation..."
        pip3 install torch
    fi

    log "Toutes les dépendances sont satisfaites."
}

# Fonction principale
main() {
    check_dependencies
    create_models_directory

    # Traiter les arguments
    if [ $# -eq 0 ]; then
        # Aucun argument, télécharger uniquement le modèle par défaut
        log "Aucun modèle spécifié, téléchargement du modèle par défaut: $DEFAULT_MODEL"
        download_model $DEFAULT_MODEL
    else
        # Traiter les options
        while [ $# -gt 0 ]; do
            case "$1" in
                -h|--help)
                    show_help
                    exit 0
                    ;;
                -a|--all)
                    log "Téléchargement de tous les modèles..."
                    for model in "${MODELS[@]}"; do
                        download_model $model
                    done
                    ;;
                -m|--model)
                    shift
                    if [ -z "$1" ]; then
                        error "Argument manquant pour l'option --model"
                    fi

                    model=$1
                    # Vérifier si le modèle est valide
                    valid=0
                    for m in "${MODELS[@]}"; do
                        if [ "$m" == "$model" ]; then
                            valid=1
                            break
                        fi
                    done

                    if [ $valid -eq 1 ]; then
                        download_model $model
                    else
                        error "Modèle invalide: $model. Options valides: ${MODELS[*]}"
                    fi
                    ;;
                *)
                    warn "Option inconnue: $1. Utilisez --help pour voir les options disponibles."
                    ;;
            esac
            shift
        done
    fi

    log "Téléchargement des modèles terminé!"
}

# Exécuter le script
main "$@"