# built-in dependencies
import traceback
from typing import Optional, Union

# 3rd party dependencies
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.commons.logger import Logger

logger = Logger()


# pylint: disable=broad-except


def represent(
    img_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    max_faces: Optional[int] = None,
):
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
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while representing: {str(err)} - {tb_str}"}, 400


def verify(
    img1_path: Union[str, np.ndarray],
    img2_path: Union[str, np.ndarray],
    model_name: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
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
        return obj
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400
def find(
    img_path: Union[str, np.ndarray],
    model_name: str,
    db_path: str,
    detector_backend: str,
    distance_metric: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
    normalization: str
):
    try:
        obj = DeepFace.find(
            img_path=img_path,
            db_path=db_path,
            model_name=model_name,
            detector_backend=detector_backend,
            normalization=normalization,
            distance_metric=distance_metric,
            align=align,
            enforce_detection=enforce_detection,
            anti_spoofing=anti_spoofing,
        )
        return obj
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while verifying: {str(err)} - {tb_str}"}, 400
def extract_faces(
    img_path: Union[str, np.ndarray],
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    color_face: str,
    expand_percentage: int,
    anti_spoofing: bool,
    grayscale: bool
):
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
        return obj
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while extracting: {str(err)} - {tb_str}"}, 400
        
def analyze(
    img_path: Union[str, np.ndarray],
    actions: list,
    detector_backend: str,
    enforce_detection: bool,
    align: bool,
    anti_spoofing: bool,
):
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
        return result
    except Exception as err:
        tb_str = traceback.format_exc()
        logger.error(str(err))
        logger.error(tb_str)
        return {"error": f"Exception while analyzing: {str(err)} - {tb_str}"}, 400
