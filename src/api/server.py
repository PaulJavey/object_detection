import cv2
import time
import threading
import queue
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable

from ..detector.model import YOLODetector
from ..detector.utils import draw_fps, filter_detections_by_confidence, filter_detections_by_classes
from .capture import VideoCapture


class VideoProcessor:
    """
    Classe pour traiter les frames vidéo et appliquer la détection d'objets.
    """

    def __init__(
            self,
            detector: YOLODetector,
            video_capture: VideoCapture,
            process_every_n_frames: int = 1,
            max_processing_fps: Optional[int] = None,
            draw_detections: bool = True,
            detection_callback: Optional[Callable] = None
    ):
        """
        Initialise le processeur vidéo.

        Args:
            detector: Instance de YOLODetector
            video_capture: Instance de VideoCapture
            process_every_n_frames: N'analyser qu'un frame sur N (économie de ressources)
            max_processing_fps: Limite le nombre de frames traités par seconde
            draw_detections: Dessiner les détections sur les frames
            detection_callback: Fonction appelée après chaque détection
        """
        self.detector = detector
        self.video_capture = video_capture
        self.process_every_n_frames = max(1, process_every_n_frames)
        self.max_processing_fps = max_processing_fps
        self.draw_detections = draw_detections
        self.detection_callback = detection_callback

        # État de traitement
        self.running = False
        self.processing_thread = None
        self.frame_count = 0
        self.processed_count = 0
        self.current_detections = []
        self.output_frame = None
        self.output_frame_lock = threading.Lock()
        self.start_time = 0

        # Buffer de sortie
        self.output_buffer = queue.Queue(maxsize=5)

    def start(self) -> bool:
        """
        Démarre le traitement vidéo.

        Returns:
            True si démarré avec succès, False sinon
        """
        # Vérifier si la capture est déjà en cours
        if not self.video_capture.is_running():
            if not self.video_capture.start():
                print("Erreur: La capture vidéo n'a pas pu être démarrée")
                return False

        # Initialiser l'état
        self.running = True
        self.frame_count = 0
        self.processed_count = 0
        self.current_detections = []
        self.start_time = time.time()

        # Démarrer le thread de traitement
        self.processing_thread = threading.Thread(target=self._processing_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()

        return True

    def stop(self) -> None:
        """
        Arrête le traitement vidéo.
        """
        self.running = False

        # Attendre que le thread s'arrête
        if self.processing_thread and self.processing_thread.is_alive():
            self.processing_thread.join(timeout=1.0)

        # Vider le buffer
        while not self.output_buffer.empty():
            try:
                self.output_buffer.get_nowait()
            except queue.Empty:
                break

    def get_processed_frame(self) -> Tuple[bool, Optional[np.ndarray]]:
        """
        Récupère le dernier frame traité avec les détections.

        Returns:
            Tuple (succès, frame)
        """
        if not self.running:
            return False, None

        try:
            frame = self.output_buffer.get_nowait()
            return True, frame
        except queue.Empty:
            # Si le buffer est vide, utiliser le dernier frame traité s'il existe
            with self.output_frame_lock:
                if self.output_frame is not None:
                    return True, self.output_frame.copy()
                return False, None

    def get_current_detections(self) -> List[Dict]:
        """
        Récupère les détections actuelles.

        Returns:
            Liste des détections
        """
        return self.current_detections.copy()

    def get_processing_fps(self) -> float:
        """
        Calcule le FPS de traitement.

        Returns:
            FPS calculé
        """
        if self.processed_count <= 0 or self.start_time == 0:
            return 0.0

        elapsed_time = time.time() - self.start_time
        return self.processed_count / elapsed_time if elapsed_time > 0 else 0.0

    def is_running(self) -> bool:
        """
        Vérifie si le traitement est en cours.

        Returns:
            True si le traitement est en cours, False sinon
        """
        return self.running

    def _processing_loop(self) -> None:
        """
        Boucle de traitement exécutée dans un thread séparé.
        Lit les frames, applique la détection et produit les frames annotés.
        """
        last_process_time = 0
        min_process_interval = 1.0 / self.max_processing_fps if self.max_processing_fps else 0

        while self.running:
            # Lire un frame
            ret, frame = self.video_capture.read()

            if not ret:
                # Pas de frame disponible
                time.sleep(0.01)
                continue

            self.frame_count += 1

            # Déterminer s'il faut traiter ce frame
            process_this_frame = (self.frame_count % self.process_every_n_frames == 0)

            # Respecter la limite de FPS pour le traitement
            current_time = time.time()
            enough_time_passed = (current_time - last_process_time) >= min_process_interval

            if process_this_frame and (min_process_interval == 0 or enough_time_passed):
                # Détecter les objets
                detections = self.detector.detect(frame)
                self.current_detections = detections

                # Mettre à jour le compteur et le timestamp
                self.processed_count += 1
                last_process_time = current_time

                # Appeler le callback si fourni
                if self.detection_callback and callable(self.detection_callback):
                    self.detection_callback(frame, detections)

            # Créer le frame de sortie (avec ou sans annotations)
            output_frame = frame.copy()

            if self.draw_detections:
                # Dessiner les détections sur le frame
                if self.current_detections:
                    for det in self.current_detections:
                        bbox = det['bbox']
                        label = det['label']
                        conf = det['confidence']

                        # Convertir en entiers
                        x1, y1, x2, y2 = map(int, bbox)

                        # Couleur différente pour chaque classe
                        color = (0, 255, 0)  # Vert par défaut

                        # Dessiner le rectangle
                        cv2.rectangle(output_frame, (x1, y1), (x2, y2), color, 2)

                        # Ajouter le label et la confiance
                        label_text = f"{label} {conf:.2f}"
                        cv2.putText(
                            output_frame,
                            label_text,
                            (x1, y1 - 10),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.5,
                            color,
                            2
                        )

                # Ajouter les FPS
                process_fps = self.get_processing_fps()
                capture_fps = self.video_capture.get_fps()

                cv2.putText(
                    output_frame,
                    f"Capture: {capture_fps:.1f} FPS, Traitement: {process_fps:.1f} FPS",
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

            # Mettre à jour le frame de sortie
            with self.output_frame_lock:
                self.output_frame = output_frame.copy()

            # Ajouter au buffer de sortie
            if self.output_buffer.full():
                try:
                    self.output_buffer.get_nowait()
                except queue.Empty:
                    pass

            try:
                self.output_buffer.put_nowait(output_frame)
            except queue.Full:
                pass

            # Petit délai pour éviter de saturer le CPU
            time.sleep(0.001)


# Pour l'importation directe
if __name__ == "__main__":
    import os

    # Test simple de la classe VideoProcessor
    try:
        # Initialiser le détecteur
        detector = YOLODetector(model_type="yolov5s")

        # Initialiser la capture vidéo
        source = 0  # Webcam
        cap = VideoCapture(source, width=640, height=480)

        # Initialiser le processeur vidéo
        processor = VideoProcessor(
            detector=detector,
            video_capture=cap,
            process_every_n_frames=2,  # Traiter un frame sur deux
            draw_detections=True
        )

        if processor.start():
            try:
                while True:
                    ret, frame = processor.get_processed_frame()

                    if not ret:
                        time.sleep(0.01)
                        continue

                    cv2.imshow("Détection d'objets", frame)

                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break

            except KeyboardInterrupt:
                print("Traitement interrompu")

            finally:
                processor.stop()
                cap.stop()
                cv2.destroyAllWindows()
        else:
            print("Échec du démarrage du processeur vidéo")

    except Exception as e:
        print(f"Erreur: {e}")