# built-in dependencies
from typing import Union

# 3rd party dependencies
from flask import Blueprint, request
import numpy as np

# project dependencies
from deepface import DeepFace
from deepface.api.src.modules.core import service
from deepface.commons import image_utils
from deepface.commons.logger import Logger

logger = Logger()

blueprint = Blueprint("routes", __name__)

# pylint: disable=no-else-return, broad-except


@blueprint.route("/")
def home():
    return f"<h1>Welcome to DeepFace API v{DeepFace.__version__}!</h1>"


def extract_image_from_request(img_key: str) -> Union[str, np.ndarray]:
    """
    Extracts an image from the request either from json or a multipart/form-data file.

    Args:
        img_key (str): The key used to retrieve the image data
            from the request (e.g., 'img1').

    Returns:
        img (str or np.ndarray): Given image detail (base64 encoded string, image path or url)
            or the decoded image as a numpy array.
    """

    # Check if the request is multipart/form-data (file input)
    if request.files:
        # request.files is instance of werkzeug.datastructures.ImmutableMultiDict
        # file is instance of werkzeug.datastructures.FileStorage
        file = request.files.get(img_key)

        if file is None:
            raise ValueError(f"Request form data doesn't have {img_key}")

        if file.filename == "":
            raise ValueError(f"No file uploaded for '{img_key}'")

        img = image_utils.load_image_from_file_storage(file)

        return img
    # Check if the request is coming as base64, file path or url from json or form data
    elif request.is_json or request.form:
        input_args = request.get_json() or request.form.to_dict()

        if input_args is None:
            raise ValueError("empty input set passed")

        # this can be base64 encoded image, and image path or url
        img = input_args.get(img_key)

        if not img:
            raise ValueError(f"'{img_key}' not found in either json or form data request")

        return img

    # If neither JSON nor file input is present
    raise ValueError(f"'{img_key}' not found in request in either json or form data")


@blueprint.route("/represent", methods=["POST"])
def represent():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    obj = service.represent(
        img_path=img,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
        max_faces=input_args.get("max_faces"),
    )

    logger.debug(obj)

    return obj


@blueprint.route("/verify", methods=["POST"])
def verify():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img1 = extract_image_from_request("img1")
    except Exception as err:
        return {"exception": str(err)}, 400

    try:
        img2 = extract_image_from_request("img2")
    except Exception as err:
        return {"exception": str(err)}, 400

    verification = service.verify(
        img1_path=img1,
        img2_path=img2,
        model_name=input_args.get("model_name", "VGG-Face"),
        detector_backend=input_args.get("detector_backend", "opencv"),
        distance_metric=input_args.get("distance_metric", "cosine"),
        align=input_args.get("align", True),
        enforce_detection=input_args.get("enforce_detection", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(verification)

    return verification
@blueprint.route("/find", methods=["POST"])
def find():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img = extract_image_from_request("img_path")
    except Exception as err:
        return {"exception": str(err)}, 400

    finder = service.find(
        img_path=img,
        db_path=input_args.get("db_path", "/app/deepface/images/my_db"),
        model_name=input_args.get("model_name", "ArcFace"),
        detector_backend=input_args.get("detector_backend", "retinaface"),
        distance_metric=input_args.get("distance_metric", "cosine"),
        align=input_args.get("align", True),
        enforce_detection=input_args.get("enforce_detection", True),
        normalization=input_args.get("normalization", "ArcFace"),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(finder)
    
    if isinstance(finder, pd.DataFrame):
        return jsonify(finder.to_dict(orient="records"))

    elif isinstance(finder, list) and all(isinstance(df, pd.DataFrame) for df in finder):
        # DeepFace may return a list of DataFrames
        all_records = []
        for df in finder:
            records = df.to_dict(orient="records")
            all_records.extend(records)  # Or append(records) if you want nested structure
        return jsonify(all_records)

    elif isinstance(finder, tuple):
        return jsonify(finder[0]), finder[1]

    else:
        try:
            return jsonify(finder)
        except TypeError as e:
            logger.error(f"Failed to jsonify object of type {type(finder)}: {str(e)}")
            return jsonify({"error": f"Cannot serialize object of type {type(finder)}"}), 500
def make_json_safe(obj):
    if isinstance(obj, dict):
        return {k: make_json_safe(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_safe(i) for i in obj]
    elif hasattr(obj, 'tolist'):  # NumPy arrays
        return obj.tolist()
    elif hasattr(obj, '__dict__'):
        return make_json_safe(vars(obj))
    elif isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    else:
        return str(obj)  # fallback for unknown types
@blueprint.route("/extract_faces", methods=["POST"])
def extract_faces():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img = extract_image_from_request("img_path")
    except Exception as err:
        return {"exception": str(err)}, 400

    extracted = service.extract_faces(
        img_path=img,       
        detector_backend=input_args.get("detector_backend", "retinaface"),
        align=input_args.get("align", True),
        grayscale=input_args.get("grayscale", False),
        color_face=input_args.get("color_face", "rgb"),
        expand_percentage=input_args.get("expand_percentage", 0),
        enforce_detection=input_args.get("enforce_detection", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(extracted)
    
    try:
        if isinstance(extracted, pd.DataFrame):
            return jsonify(make_json_safe(extracted.to_dict(orient="records")))

        elif isinstance(extracted, list) and all(isinstance(df, pd.DataFrame) for df in extracted):
            all_records = []
            for df in extracted:
                records = df.to_dict(orient="records")
                all_records.extend(records)
            return jsonify(make_json_safe(all_records))

        elif isinstance(extracted, tuple):
            return jsonify(make_json_safe(extracted[0])), extracted[1]

        else:
            return jsonify(make_json_safe(extracted))

    except TypeError as e:
        logger.error(f"Failed to jsonify object of type {type(extracted)}: {str(e)}")
        return jsonify({"error": f"Cannot serialize object of type {type(extracted)}"}), 500
@blueprint.route("/analyze", methods=["POST"])
def analyze():
    input_args = (request.is_json and request.get_json()) or (
        request.form and request.form.to_dict()
    )

    try:
        img = extract_image_from_request("img")
    except Exception as err:
        return {"exception": str(err)}, 400

    actions = input_args.get("actions", ["age", "gender", "emotion", "race"])
    # actions is the only argument instance of list or tuple
    # if request is form data, input args can either be text or file
    if isinstance(actions, str):
        actions = (
            actions.replace("[", "")
            .replace("]", "")
            .replace("(", "")
            .replace(")", "")
            .replace('"', "")
            .replace("'", "")
            .replace(" ", "")
            .split(",")
        )

    demographies = service.analyze(
        img_path=img,
        actions=actions,
        detector_backend=input_args.get("detector_backend", "opencv"),
        enforce_detection=input_args.get("enforce_detection", True),
        align=input_args.get("align", True),
        anti_spoofing=input_args.get("anti_spoofing", False),
    )

    logger.debug(demographies)

    return demographies
