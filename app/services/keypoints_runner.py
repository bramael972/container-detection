import cv2
import os
import numpy as np
import matplotlib.pyplot as plt
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import yaml

class PredictorKeySingleton:
    _instance = None

    @staticmethod
    def get_instance():
        if PredictorKeySingleton._instance is None:
            print("➡️ Creating predictor keys...")
            PredictorKeySingleton._instance = PredictorKeySingleton._create_predictor()
        print("✅ Returning predictor instance keys:", PredictorKeySingleton._instance)
        return PredictorKeySingleton._instance

    @staticmethod
    def _create_predictor():
        # Charger la configuration YAML
        with open("app/config/config_keypoints.yaml", "r") as f:
            cfg_dict = yaml.safe_load(f)

        dataset_cfg = cfg_dict["dataset"]
        model_cfg = cfg_dict["model"]

        base_path = dataset_cfg["base_path"]
        train_json = os.path.join(base_path, dataset_cfg["train_json"])
        val_json = os.path.join(base_path, dataset_cfg["val_json"])
        train_images = os.path.join(base_path, dataset_cfg["train_images"])
        val_images = os.path.join(base_path, dataset_cfg["val_images"])

        # Enregistrement des datasets
        register_coco_instances("my_dataset_train_keypoint", {}, train_json, train_images)
        register_coco_instances("my_dataset_val_keypoint", {}, val_json, val_images)

        keypoint_names = dataset_cfg["keypoint_names"]
        keypoint_flip_map = [tuple(pair) for pair in dataset_cfg["keypoint_flip_map"]]

        for d in ["my_dataset_train_keypoint", "my_dataset_val_keypoint"]:
            meta = MetadataCatalog.get(d)
            meta.keypoint_names = keypoint_names
            meta.keypoint_flip_map = keypoint_flip_map

        cfg = get_cfg()
        cfg.OUTPUT_DIR = model_cfg["output_dir"]
        cfg.merge_from_file(model_cfg["config_file"])
        cfg.DATASETS.TRAIN = ("my_dataset_train_keypoint",)
        cfg.MODEL.WEIGHTS = model_cfg["weights_path"]
        cfg.SOLVER.IMS_PER_BATCH = model_cfg["ims_per_batch"]
        cfg.SOLVER.BASE_LR = model_cfg["base_lr"]
        cfg.SOLVER.MAX_ITER = model_cfg["max_iter"]
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = model_cfg["batch_size_per_image"]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = model_cfg["num_classes"]
        cfg.MODEL.KEYPOINT_ON = True
        cfg.MODEL.ROI_KEYPOINT_HEAD.NUM_KEYPOINTS = model_cfg["num_keypoints"]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_cfg["score_thresh_test"]
        cfg.MODEL.DEVICE = model_cfg.get("device", "cuda")

        return DefaultPredictor(cfg)

def process_model_kpt_extract(image_path, output_dir="app/output_image_kpt_model", score_thresh=1):
    os.makedirs(output_dir, exist_ok=True)
    predictor = PredictorKeySingleton.get_instance()
    im = cv2.imread(image_path)
    # Définir la liste des noms de keypoints dans l'ordre utilisé dans CVAT
    keypoint_names = [
        "x1_t",
        "x2_t",
        "x3_t",
        "x4_t",
        "x1_b",
        "x2_b",
        "x3_b",
        "x4_b",
    ]

    # # Créer une table de correspondance pour flip horizontal
    # keypoint_flip_map = [(0, 1), (2, 3), (4, 5), (6, 7)]

    # # Injecter ces métadonnées dans le MetadataCatalog
    # MetadataCatalog.get("my_dataset_train_keypoint").keypoint_names = keypoint_names
    # MetadataCatalog.get("my_dataset_train_keypoint").keypoint_flip_map = keypoint_flip_map

    # # Idem pour validation si tu en as une
    # MetadataCatalog.get("my_dataset_val_keypoint").keypoint_names = keypoint_names
    # MetadataCatalog.get("my_dataset_val_keypoint").keypoint_flip_map = keypoint_flip_map

    if im is None:
        raise ValueError(f"Impossible de charger l'image : {image_path}")

    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")

    if not instances.has("pred_keypoints"):
        print("Aucun keypoint détecté.")
        return {
            "method": "detectronKeyPoint",
            "points2D": [],
            "img": os.path.basename(image_path),
            "point3D": None,
            "param_opti": None,
            "iou":None
        }

    keypoints = instances.pred_keypoints
    keypoint_names = MetadataCatalog.get("my_dataset_train_keypoint").keypoint_names
    img_vis = im.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    points2D = []
    for i in range(len(keypoints)):
        for j, kp in enumerate(keypoints[i]):
            x, y, v = kp
            print(v)
            if v > score_thresh:
                cv2.circle(img_vis, (int(x), int(y)), radius=8, color=(0, 0, 255), thickness=-1)
                cv2.putText(
                    img_vis,
                    keypoint_names[j],
                    (int(x), int(y) - 10),
                    font,
                    0.7,
                    (0, 0, 255),
                    thickness=2
                )
                points2D.append({
                    "name": keypoint_names[j],
                    "x": float(x),
                    "y": float(y),
                    "score": float(v)
                })

    # Sauvegarde finale
    base_filename = os.path.basename(image_path).split(".")[0]
    output_path = os.path.join(output_dir, f"{base_filename}_all_keypoints.png")
    # plt.figure(figsize=(10, 8))
    # plt.imshow(cv2.cvtColor(img_vis, cv2.COLOR_BGR2RGB))
    # plt.axis("off")
    # plt.tight_layout()
    # plt.savefig(output_path, bbox_inches="tight", pad_inches=0.1)
    # plt.close()
    # output_path = os.path.join(output_dir, f"{base_filename}_all_keypoints.png")
    cv2.imwrite(output_path, img_vis)

    return {
        "method": "detectronKeyPoint",
        "points2D": points2D,
        "img": os.path.basename(image_path),
        "point3D": None,
        "param_opti": None,
        "iou":None
    }