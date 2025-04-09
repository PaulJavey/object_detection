from fastapi import APIRouter, File, UploadFile, HTTPException, Depends, Query
from fastapi.responses import JSONResponse, StreamingResponse
from typing import List, Dict, Any, Optional
import cv2
import numpy as np
import tempfile
import os
import time
import asyncio
from pydantic import BaseModel

from ..detector.model import YOLODetector
from ..video.capture import VideoCapture
from ..video.processor import VideoProcessor

# Créer le routeur
router = APIRouter(tags=["detection"])


# Modèles de données
class DetectionResult(BaseModel):
    class_id: int
    label: str
    confidence: float
    bbox: List[float]


class DetectionResponse(BaseModel):
    detections: List[DetectionResult]
    processing_time: float
    image_size: Dict[str, int]


class VideoStreamOptions(BaseModel):
    source: Optional[str] = None
    width: Optional[int] = 640
    height: Optional[int] = 480
    fps: Optional[int] = 30


# Dépendance pour obtenir le détecteur
async def get_detector():
    # Normalement ce serait un singleton ou géré par un gestionnaire de dépendances
    # Pour l'exemple, on crée une nouvelle instance
    detector = YOLODetector(model_type="yolov5s")
    try:
        yield detector
    finally:
        # Nettoyage éventuel
        pass


@router.post("/detect", response_model=DetectionResponse)
async def detect_objects(
        file: UploadFile = File(...),
        detector: YOLODetector = Depends(get_detector),
        min_confidence: float = Query(0.45, ge=0, le=1)
):
    """
    Détecte les objets dans une image téléchargée.

    - **file**: Fichier image à analyser (jpg, png)
    - **min_confidence**: Seuil de confiance minimum (0-1)
    """
    try:
        # Lire et valider l'image
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(contents)
            temp_path = temp.name

        image = cv2.imread(temp_path)
        os.unlink(temp_path)

        if image is None:
            raise HTTPException(status_code=400, detail="Image invalide ou format non pris en charge")

        # Configurer le seuil de confiance
        detector.confidence_threshold = min_confidence

        # Effectuer la détection
        start_time = time.time()
        detections = detector.detect(image)
        processing_time = time.time() - start_time

        # Préparer la réponse
        detection_results = []
        for det in detections:
            detection_results.append(
                DetectionResult(
                    class_id=det["class"],
                    label=det["label"],
                    confidence=det["confidence"],
                    bbox=det["bbox"]
                )
            )

        return DetectionResponse(
            detections=detection_results,
            processing_time=processing_time,
            image_size={"width": image.shape[1], "height": image.shape[0]}
        )

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la détection: {str(e)}")


@router.post("/detect/with-visualization")
async def detect_with_visualization(
        file: UploadFile = File(...),
        detector: YOLODetector = Depends(get_detector),
        min_confidence: float = Query(0.45, ge=0, le=1),
        draw_labels: bool = Query(True),
        draw_confidence: bool = Query(True)
):
    """
    Détecte les objets et renvoie l'image annotée.

    - **file**: Fichier image à analyser
    - **min_confidence**: Seuil de confiance minimum (0-1)
    - **draw_labels**: Afficher les étiquettes de classe
    - **draw_confidence**: Afficher les scores de confiance
    """
    try:
        # Lire et valider l'image
        contents = await file.read()
        with tempfile.NamedTemporaryFile(delete=False) as temp:
            temp.write(contents)
            temp_path = temp.name

        image = cv2.imread(temp_path)
        os.unlink(temp_path)

        if image is None:
            raise HTTPException(status_code=400, detail="Image invalide ou format non pris en charge")

        # Configurer le seuil de confiance
        detector.confidence_threshold = min_confidence

        # Détecter et annoter
        result_image = detector.detect_and_draw(
            image,
            draw_labels=draw_labels,
            draw_confidence=draw_confidence
        )

        # Convertir en JPEG
        _, buffer = cv2.imencode('.jpg', result_image)
        io_buf = io.BytesIO(buffer)

        return StreamingResponse(io_buf, media_type="image/jpeg")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors de la détection: {str(e)}")


@router.post("/stream/start")
async def start_video_stream(
        options: VideoStreamOptions,
        detector: YOLODetector = Depends(get_detector)
):
    """
    Démarre un flux vidéo avec détection d'objets.

    - **options**: Options de configuration du flux vidéo
    """
    # Dans une application réelle, vous stockeriez l'état dans une base de données
    # ou un gestionnaire d'état pour permettre de récupérer le flux plus tard
    try:
        # Créer les instances de capture et traitement vidéo
        video_capture = VideoCapture(
            source=options.source if options.source else 0,
            width=options.width,
            height=options.height,
            fps=options.fps
        )

        processor = VideoProcessor(
            detector=detector,
            video_capture=video_capture,
            draw_detections=True
        )

        # Démarrer le traitement
        if processor.start():
            # Stocker les informations du flux
            stream_id = "stream_" + str(int(time.time()))

            return {
                "status": "success",
                "stream_id": stream_id,
                "message": "Flux vidéo démarré avec succès",
                "stream_url": f"/stream/{stream_id}"
            }
        else:
            raise HTTPException(status_code=500, detail="Échec du démarrage du flux vidéo")

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erreur lors du démarrage du flux: {str(e)}")


@router.get("/stream/{stream_id}")
async def get_video_stream(stream_id: str):
    """
    Récupère un flux vidéo en cours avec détections.

    - **stream_id**: Identifiant du flux vidéo
    """
    # Dans une implémentation réelle, vous récupéreriez le processeur
    # à partir d'un gestionnaire d'état en utilisant stream_id

    # Exemple simplifié : nous simulons un flux générique
    return StreamingResponse(
        generate_mock_stream(),
        media_type="multipart/x-mixed-replace;boundary=frame"
    )


async def generate_mock_stream():
    """Générateur de frames simulé pour l'exemple"""
    # Dans une implémentation réelle, vous utiliseriez le processeur vidéo réel
    mock_image = np.zeros((480, 640, 3), dtype=np.uint8)

    for i in range(100):  # Limité à 100 frames pour l'exemple
        # Dessiner quelque chose sur l'image pour simuler un changement
        cv2.putText(
            mock_image,
            f"Frame {i}",
            (50, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

        # Encoder en JPEG
        ret, buffer = cv2.imencode('.jpg', mock_image)
        frame_bytes = buffer.tobytes()

        # Retourner le frame
        yield (
                b'--frame\r\n'
                b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n'
        )

        await asyncio.sleep(0.1)  # 10 FPS simulés
