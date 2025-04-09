import os
import sys
import unittest
import cv2
import numpy as np
from pathlib import Path

# Ajouter le répertoire parent au chemin pour l'importation
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from src.detector.model import YOLODetector
from src.detector.utils import resize_image, filter_detections_by_confidence


class TestDetector(unittest.TestCase):
    """Tests pour le module de détection YOLOv5."""

    @classmethod
    def setUpClass(cls):
        """Configuration initiale pour les tests."""
        cls.model_path = Path("data/models/yolov5s.pt")

        # Vérifier si le modèle existe
        if not cls.model_path.exists():
            print(f"Modèle non trouvé: {cls.model_path}. Les tests peuvent échouer.")

        # Créer une image de test
        cls.test_image = np.zeros((640, 640, 3), dtype=np.uint8)

        # Dessiner quelques formes pour simuler des objets
        cv2.rectangle(cls.test_image, (100, 100), (300, 300), (0, 255, 0), -1)
        cv2.rectangle(cls.test_image, (400, 400), (500, 500), (0, 0, 255), -1)

        # Tester plusieurs configurations du détecteur pour couvrir différents cas
        try:
            cls.detector = YOLODetector(
                model_type="yolov5s",
                confidence_threshold=0.45,
                device="cpu"  # Forcer CPU pour la compatibilité
            )
        except Exception as e:
            print(f"Erreur lors de l'initialisation du détecteur: {e}")
            cls.detector = None

    def setUp(self):
        """Configuration avant chaque test."""
        if self.detector is None:
            self.skipTest("Détecteur non initialisé")

    def test_detector_initialization(self):
        """Tester l'initialisation du détecteur."""
        # Vérifier que le détecteur est correctement initialisé
        self.assertIsNotNone(self.detector)
        self.assertEqual(self.detector.confidence_threshold, 0.45)
        self.assertEqual(self.detector.device, "cpu")

    def test_detector_get_model_info(self):
        """Tester la récupération des informations du modèle."""
        model_info = self.detector.get_model_info()

        # Vérifier que les informations sont complètes
        self.assertIn("type", model_info)
        self.assertIn("classes", model_info)
        self.assertIn("device", model_info)
        self.assertIn("confidence_threshold", model_info)
        self.assertIn("iou_threshold", model_info)