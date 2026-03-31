# built-in dependencies
import traceback
import os
import pickle
from typing import Optional, Union, Dict, Any, Tuple, List

# force TF to CPU - leave GPU for ONNX/InsightFace
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')

# 3rd party dependencies
import numpy as np
import cv2
from numpy.typing import NDArray
from insightface.app import FaceAnalysis

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()

# Initialise InsightFace once at module load - ONNX, lean memory footprint
_face_app = FaceAnalysis(
    providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
)
_face_app.prepare(ctx_id=0, det_size=(640, 640))


# pylint: disable=broad-except, too-many-positional-arguments


def _load_image_to_numpy(img_path: Union[str, np.ndarray]) -> np.ndarray:
    if isinstance(img_path, np.ndarray):
        return img_path
    if isinstance(img_path, str) and img_path.startswith("data:image"):
        import base64
        header, data = img_path.split(",", 1)
        decoded = base64.b64decode(data)
        nparr = np.frombuffer(decoded, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    return cv2.imread(img_path)


def _build_db_embeddings(db_path: str, force_rebuild: bool = False):
    cache_file = os.path.join(db_path, "embeddings.pkl")

    if os.path.exists(cache_file) and not force_rebuild:
        # auto-rebuild if any image is newer than cache
        cache_mtime = os.path.getmtime(cache_file)
        needs_rebuild = any(
            os.path.getmtime(os.path.join(db_path, p, f)) > cache_mtime
            for p in os.listdir(db_path)
            if os.path.isdir(os.path.join(db_path, p))  # guard here
            for f in os.listdir(os.path.join(db_path, p))
        )
        if not needs_rebuild:
            with open(cache_file, "rb") as f:
                return pickle.load(f)

    db = {}
    for person_name in os.listdir(db_path):
        person_dir = os.path.join(db_path, person_name)
        if not os.path.isdir(person_dir):
            continue
        embeddings = []
        for img_file in os.listdir(person_dir):
            img_fp = os.path.join(person_dir, img_file)
            img = cv2.imread(img_fp)
            if img is None:
                continue
            faces = _face_app.get(img)
            if faces:
                embeddings.append(faces[0].normed_embedding)
        if embeddings:  # only add if we got valid embeddings
            db[person_name] = embeddings

    with open(cache_file, "wb") as f:
        pickle.dump(db, f)

    logger.info(f"Built embedding DB with {len(db)} identities from {db_path}")
    return db


def _cosine_distance(a, b):
    return 1 - np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def represent(
    img_path: Union[str, NDArray[Any]],
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
) -> Tuple[Dict[str, Any], int]:
    try:
        result = {}
        embedding_objs = DeepFace.represent(
            img_path=img_path,
            model_name=model_name,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            anti_spoofing=anti_spoofing,
            max_faces=max_faces,
        )
        result["results"] = embedding_objs
        return result, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: Union[str, NDArray[Any]],
    img2_path: Union[str, NDArray[Any]],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
) -> Tuple[Dict[str, Any], int]:
    try:
        obj = DeepFace.verify(
            img1_path=img1_path,
            img2_path=img2_path,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return obj, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400


def analyze(
    img_path: Union[str, NDArray[Any]],
    actions: List[str],
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
) -> Tuple[Dict[str, Any], int]:
    try:
        result = {}
        demographies = DeepFace.analyze(
            img_path=img_path,
            actions=actions,
            detector_backend=detector_backend,
            enforce_detection=enforce_detection,
            align=align,
            silent=True,
            anti_spoofing=anti_spoofing,
        )
        result["results"] = demographies
        return result, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400


def find(
    img_path: Union[str, np.ndarray],
    model_name: str,
    db_path: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    normalization: str,
    threshold: float = 0.4,
) -> List[Dict[str, Any]]:
    try:
        img = _load_image_to_numpy(img_path)
        if img is None:
            return {"error": "Could not load image"}, 400

        faces = _face_app.get(img)
        if not faces:
            if enforce_detection:
                return {"error": "No face detected"}, 400
            return []

        db = _build_db_embeddings(db_path)
        if not db:
            return {"error": f"No embeddings found in {db_path}"}, 400

        results = []
        for face in faces:
            query_embedding = face.normed_embedding
            best_match = None
            best_distance = float("inf")

            for person_name, embeddings in db.items():
                for emb in embeddings:
                    dist = _cosine_distance(query_embedding, emb)
                    if dist < best_distance:
                        best_distance = dist
                        best_match = person_name

            results.append({
                "identity": best_match if best_distance < threshold else "unknown",
                "distance": float(best_distance),
                "threshold": threshold,
                "verified": best_distance < threshold,
                "bbox": face.bbox.tolist(),
            })

        return results

    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while finding: {str(err)} - {tb_str}"}, 400


def extract_faces(
    img_path: Union[str, np.ndarray],
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    color_face: str,
    expand_percentage: int,
    anti_spoofing: bool,
    grayscale: bool,
) -> Tuple[Any, int]:
    try:
        obj = DeepFace.extract_faces(
            img_path=img_path,
            detector_backend=detector_backend,
            expand_percentage=expand_percentage,
            align=align,
            color_face=color_face,
            enforce_detection=enforce_detection,
            grayscale=grayscale,
            anti_spoofing=anti_spoofing,
        )
        return obj, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while extracting: {str(err)} - {tb_str}"}, 400


def register(
    img: Union[str, NDArray[Any]],
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    l2_normalize: bool,
    expand_percentage: int,
    normalization: str,
    anti_spoofing: bool,
    img_name: Optional[str],
    database_type: str,
    connection_details: str,
) -> Tuple[Dict[str, Any], int]:
    try:
        return (
            DeepFace.register(
                img=img,
                img_name=img_name,
                model_name=model_name,
                detector_backend=detector_backend,
                enforce_detection=enforce_detection,
                align=align,
                l2_normalize=l2_normalize,
                expand_percentage=expand_percentage,
                normalization=normalization,
                anti_spoofing=anti_spoofing,
                database_type=database_type,
                connection_details=connection_details,
            ),
            200,
        )
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while registering: {str(err)} - {tb_str}"}, 400


def search(
    img: Union[str, NDArray[Any]],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    l2_normalize: bool,
    expand_percentage: int,
    normalization: str,
    anti_spoofing: bool,
    similarity_search: bool,
    k: Optional[int],
    database_type: str,
    connection_details: str,
    search_method: str,
) -> Tuple[Dict[str, Any], int]:
    try:
        result = {}
        dfs = DeepFace.search(
            img=img,
            model_name=model_name,
            detector_backend=detector_backend,
            distance_metric=distance_metric,
            enforce_detection=enforce_detection,
            align=align,
            l2_normalize=l2_normalize,
            expand_percentage=expand_percentage,
            normalization=normalization,
            anti_spoofing=anti_spoofing,
            similarity_search=similarity_search,
            k=k,
            database_type=database_type,
            connection_details=connection_details,
            search_method=search_method,
        )
        result["results"] = [df.to_dict(orient="records") for df in dfs]
        return result, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while searching: {str(err)} - {tb_str}"}, 400


def build_index(
    model_name: str,
    detector_backend: str,
    align: bool,
    l2_normalize: bool,
    database_type: str,
    connection_details: str,
) -> Tuple[Dict[str, Any], int]:
    try:
        DeepFace.build_index(
            model_name=model_name,
            detector_backend=detector_backend,
            align=align,
            l2_normalize=l2_normalize,
            database_type=database_type,
            connection_details=connection_details,
        )
        return {"message": "Index built successfully"}, 200
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while building index: {str(err)} - {tb_str}"}, 400
