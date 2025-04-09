import os
import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union


class YOLODetector:
    """
    Classe pour la détection d'objets utilisant YOLOv5.
    """

    def __init__(
            self,
            model_type: str = "yolov5s",
            confidence_threshold: float = 0.45,
            iou_threshold: float = 0.45,
            device: str = None,
            classes: Optional[List[int]] = None
    ):
        """
        Initialise le détecteur YOLOv5.

        Args:
            model_type: Type de modèle YOLOv5 ('yolov5n', 'yolov5s', 'yolov5m', 'yolov5l', 'yolov5x')
            confidence_threshold: Seuil de confiance pour les détections
            iou_threshold: Seuil IOU pour la suppression non-maximale
            device: Appareil pour l'inférence ('cpu', 'cuda:0', etc.)
            classes: Liste des indices de classes à détecter (None pour toutes)
        """
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.classes = classes

        # Déterminer l'appareil à utiliser
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        print(f"Utilisation de l'appareil: {self.device}")

        # Charger le modèle YOLOv5
        try:
            self.model = torch.hub.load('ultralytics/yolov5', model_type, pretrained=True)
            self.model.to(self.device)
            self.model.conf = confidence_threshold
            self.model.iou = iou_threshold
            self.model.classes = classes
            self.model.eval()
            print(f"Modèle {model_type} chargé avec succès")
        except Exception as e:
            print(f"Erreur lors du chargement du modèle: {e}")
            raise

    def detect(
            self,
            image: np.ndarray,
            size: int = 640
    ) -> List[Dict]:
        """
        Détecte les objets dans une image.

        Args:
            image: Image NumPy (BGR ou RGB)
            size: Taille de redimensionnement pour l'inférence

        Returns:
            Liste des détections avec format:
            [
                {
                    'class': id de classe,
                    'label': nom de classe,
                    'confidence': score de confiance,
                    'bbox': [x1, y1, x2, y2]
                },
                ...
            ]
        """
        # Préparation de l'image
        img = image.copy()

        # Inférence
        results = self.model(img, size=size)

        # Extraction des résultats
        detections = []

        # Accéder aux détections brutes
        pred = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, conf, cls]

        for det in pred:
            x1, y1, x2, y2, conf, cls = det
            cls = int(cls)

            if self.classes is not None and cls not in self.classes:
                continue

            detection = {
                'class': cls,
                'label': results.names[cls],
                'confidence': float(conf),
                'bbox': [float(x1), float(y1), float(x2), float(y2)]
            }

            detections.append(detection)

        return detections

    def detect_and_draw(
            self,
            image: np.ndarray,
            draw_labels: bool = True,
            draw_confidence: bool = True,
            line_thickness: int = 2
    ) -> np.ndarray:
        """
        Détecte les objets et dessine les résultats sur l'image.

        Args:
            image: Image à traiter
            draw_labels: Afficher les étiquettes de classe
            draw_confidence: Afficher les scores de confiance
            line_thickness: Épaisseur des lignes de bounding box

        Returns:
            Image avec annotations
        """
        img_copy = image.copy()
        detections = self.detect(img_copy)

        # Dessiner les détections
        for det in detections:
            bbox = det['bbox']
            label = det['label']
            conf = det['confidence']

            # Coordonnées du rectangle
            x1, y1, x2, y2 = map(int, bbox)

            # Couleur en fonction de la classe (pour la consistance)
            color = self._get_color_for_class(det['class'])

            # Dessiner le rectangle
            cv2.rectangle(img_copy, (x1, y1), (x2, y2), color, line_thickness)

            # Ajouter l'étiquette
            if draw_labels:
                label_text = f"{label}"
                if draw_confidence:
                    label_text += f" {conf:.2f}"

                # Position et taille du texte
                t_size = cv2.getTextSize(label_text, 0, 0.5, 1)[0]

                # Rectangle pour le texte
                cv2.rectangle(
                    img_copy,
                    (x1, y1 - t_size[1] - 3),
                    (x1 + t_size[0], y1),
                    color,
                    -1
                )

                # Texte
                cv2.putText(
                    img_copy,
                    label_text,
                    (x1, y1 - 2),
                    0,
                    0.5,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA
                )

        return img_copy

    def _get_color_for_class(self, class_id: int) -> Tuple[int, int, int]:
        """
        Génère une couleur consistante pour une classe.

        Args:
            class_id: ID de la classe

        Returns:
            Tuple RGB
        """
        colors = [
            (0, 255, 0),  # Vert
            (0, 0, 255),  # Rouge
            (255, 0, 0),  # Bleu
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Jaune
            (255, 0, 255),  # Magenta
            (192, 192, 192),  # Gris clair
            (128, 128, 128),  # Gris
            (128, 0, 0),  # Bordeaux
            (0, 128, 0),  # Vert foncé
        ]

        return colors[class_id % len(colors)]

    def get_model_info(self) -> Dict:
        """
        Renvoie les informations sur le modèle.

        Returns:
            Dictionnaire d'informations
        """
        return {
            "type": self.model.type,
            "classes": self.model.names,
            "device": self.device,
            "confidence_threshold": self.confidence_threshold,
            "iou_threshold": self.iou_threshold
        }


# Pour l'importation directe
if __name__ == "__main__":
    import cv2

    # Exemple d'utilisation
    detector = YOLODetector(model_type="yolov5s")

    # Chargement d'une image de test
    image_path = "data/samples/sample_image.jpg"
    if os.path.exists(image_path):
        image = cv2.imread(image_path)

        # Détection
        detections = detector.detect(image)
        print(f"Détections: {detections}")

        # Dessin des détections
        result_img = detector.detect_and_draw(image)

        # Affichage
        cv2.imshow("Détections", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"Image de test non trouvée: {image_path}")
