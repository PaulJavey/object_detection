import cv2
import numpy as np
import time
from typing import List, Dict, Tuple, Union, Optional


def resize_image(
        image: np.ndarray,
        target_size: Tuple[int, int] = (640, 640),
        keep_ratio: bool = True
) -> np.ndarray:
    """
    Redimensionne une image tout en conservant le ratio d'aspect si nécessaire.

    Args:
        image: Image à redimensionner
        target_size: Taille cible (largeur, hauteur)
        keep_ratio: Conserver le ratio d'aspect

    Returns:
        Image redimensionnée
    """
    h, w = image.shape[:2]
    th, tw = target_size

    if keep_ratio:
        # Calculer le ratio de mise à l'échelle
        scale = min(th / h, tw / w)
        new_h, new_w = int(h * scale), int(w * scale)

        # Redimensionner l'image
        resized = cv2.resize(image, (new_w, new_h))

        # Créer une image noire de taille cible
        new_image = np.zeros((th, tw, 3), dtype=np.uint8)

        # Centrer l'image redimensionnée
        y_offset = (th - new_h) // 2
        x_offset = (tw - new_w) // 2

        # Copier l'image redimensionnée sur l'image noire
        new_image[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        return new_image
    else:
        # Redimensionner directement
        return cv2.resize(image, (tw, th))


def convert_detections_to_absolute(
        detections: List[Dict],
        original_size: Tuple[int, int],
        input_size: Tuple[int, int]
) -> List[Dict]:
    """
    Convertit les coordonnées des détections relatives à input_size
    en coordonnées absolues pour original_size.

    Args:
        detections: Liste des détections
        original_size: Taille originale de l'image (largeur, hauteur)
        input_size: Taille d'entrée du modèle (largeur, hauteur)

    Returns:
        Liste des détections avec coordonnées ajustées
    """
    orig_w, orig_h = original_size
    input_w, input_h = input_size

    # Calculer les facteurs d'échelle
    scale_x = orig_w / input_w
    scale_y = orig_h / input_h

    adjusted_detections = []

    for det in detections:
        # Copier la détection
        adjusted_det = det.copy()

        # Obtenir les coordonnées
        x1, y1, x2, y2 = det['bbox']

        # Ajuster les coordonnées
        adjusted_det['bbox'] = [
            int(x1 * scale_x),
            int(y1 * scale_y),
            int(x2 * scale_x),
            int(y2 * scale_y)
        ]

        adjusted_detections.append(adjusted_det)

    return adjusted_detections


def filter_detections_by_classes(
        detections: List[Dict],
        classes: List[int]
) -> List[Dict]:
    """
    Filtre les détections pour ne conserver que certaines classes.

    Args:
        detections: Liste des détections
        classes: Liste des IDs de classe à conserver

    Returns:
        Liste filtrée des détections
    """
    return [det for det in detections if det['class'] in classes]


def filter_detections_by_confidence(
        detections: List[Dict],
        threshold: float
) -> List[Dict]:
    """
    Filtre les détections en fonction du seuil de confiance.

    Args:
        detections: Liste des détections
        threshold: Seuil de confiance minimum

    Returns:
        Liste filtrée des détections
    """
    return [det for det in detections if det['confidence'] >= threshold]


def calculate_fps(
        start_time: float,
        frame_count: int,
        window_size: int = 30
) -> float:
    """
    Calcule les FPS sur une fenêtre glissante.

    Args:
        start_time: Temps de départ
        frame_count: Nombre de frames traités
        window_size: Taille de la fenêtre pour le calcul

    Returns:
        FPS calculés
    """
    if frame_count <= 0:
        return 0.0

    elapsed_time = time.time() - start_time

    if frame_count < window_size:
        # Calcul sur tous les frames si moins que window_size
        return frame_count / elapsed_time
    else:
        # Calcul sur la fenêtre glissante
        return window_size / (elapsed_time * window_size / frame_count)


def draw_fps(
        image: np.ndarray,
        fps: float,
        position: Tuple[int, int] = (10, 30),
        font_scale: float = 1.0,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 2
) -> np.ndarray:
    """
    Dessine le FPS sur l'image.

    Args:
        image: Image sur laquelle dessiner
        fps: Valeur FPS à afficher
        position: Position du texte (x, y)
        font_scale: Échelle de la police
        color: Couleur du texte (B, G, R)
        thickness: Épaisseur du texte

    Returns:
        Image avec FPS affiché
    """
    img_copy = image.copy()
    text = f"FPS: {fps:.1f}"

    cv2.putText(
        img_copy,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        font_scale,
        color,
        thickness,
        cv2.LINE_AA
    )

    return img_copy


def crop_detection(
        image: np.ndarray,
        bbox: List[float],
        margin: int = 0
) -> np.ndarray:
    """
    Extrait une zone de détection de l'image.

    Args:
        image: Image source
        bbox: Coordonnées [x1, y1, x2, y2]
        margin: Marge supplémentaire autour de la détection

    Returns:
        Image recadrée
    """
    h, w = image.shape[:2]
    x1, y1, x2, y2 = map(int, bbox)

    # Ajouter une marge
    x1 = max(0, x1 - margin)
    y1 = max(0, y1 - margin)
    x2 = min(w, x2 + margin)
    y2 = min(h, y2 + margin)

    return image[y1:y2, x1:x2].copy()