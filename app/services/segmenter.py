import cv2
import os
import numpy as np
from detectron2.data.datasets import register_coco_instances
from detectron2.utils.visualizer import Visualizer, ColorMode
from detectron2.data import MetadataCatalog, DatasetCatalog
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
import requests
from scipy.optimize import minimize
from sklearn.linear_model import LinearRegression
from azure.storage.blob import BlobServiceClient
import yaml

class PredictorSingleton:
    _instance = None


    @staticmethod
    def get_instance():
        if PredictorSingleton._instance is None:
            print("➡️ Creating predictor...")
            PredictorSingleton._instance = PredictorSingleton._create_predictor()
        print("✅ Returning predictor instance:", PredictorSingleton._instance)
        return PredictorSingleton._instance

    @staticmethod
    def _create_predictor():
        with open("app/config/config_seg.yaml", "r") as f:
            cfg_data = yaml.safe_load(f)

        dataset_cfg = cfg_data["dataset"]
        model_cfg = cfg_data["model"]
        train_cfg = cfg_data["training"]

        register_coco_instances(dataset_cfg["name"], {}, dataset_cfg["annotations"], dataset_cfg["images"])
        metadata = MetadataCatalog.get(dataset_cfg["name"])
        dataset_dicts = DatasetCatalog.get(dataset_cfg["name"])

        cfg = get_cfg()
        cfg.OUTPUT_DIR = model_cfg["output_dir"]
        cfg.merge_from_file(model_cfg["config_file"])
        cfg.DATASETS.TRAIN = (dataset_cfg["name"],)
        cfg.DATALOADER.NUM_WORKERS = train_cfg["num_workers"]
        cfg.MODEL.WEIGHTS = model_cfg["weights"]
        cfg.SOLVER.IMS_PER_BATCH = train_cfg["ims_per_batch"]
        cfg.SOLVER.BASE_LR = train_cfg["base_lr"]
        cfg.SOLVER.MAX_ITER = train_cfg["max_iter"]
        cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = train_cfg["batch_size_per_image"]
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = model_cfg["num_classes"]
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = model_cfg["score_thresh_test"]
        cfg.MODEL.DEVICE = model_cfg.get("device", "cuda")
        return DefaultPredictor(cfg)

def download_image(image_url: str):
    print(image_url)
    response = requests.get(image_url)
    
    if response.status_code != 200:
        raise Exception(f"Failed to download image. HTTP status code: {response.status_code}")
    
    image_array = np.asarray(bytearray(response.content), dtype=np.uint8)
    
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    
    if image is None:
        raise Exception("Failed to decode image")
    
    return image


def process_image(image_path, output_dir="app/output_cropped",online=False, score_thresh=0.6):
    predictor = PredictorSingleton.get_instance()
    if online:
        im = download_image(image_path)
    else:
        im = cv2.imread(image_path)
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")

    boxes = instances.pred_boxes
    scores = instances.scores
    classes = instances.pred_classes
    masks = instances.pred_masks
    poubelle_class_id = MetadataCatalog.get("my_dataset_train").thing_classes.index("poubelle")
    os.makedirs(output_dir, exist_ok=True)
    saved_files = []

    for i in range(len(classes)):
        if classes[i] == poubelle_class_id and scores[i] >= score_thresh:
            poubelle_mask = masks[i].numpy()

            # Vérifier si un autre masque est entièrement contenu
            for j in range(len(classes)):
                if i != j and masks[j].numpy().sum() > 0:
                    intersection = np.logical_and(poubelle_mask, masks[j].numpy())
                    if np.all(intersection == masks[j].numpy()):
                        pass

            masked_image = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.uint8)

            for c in range(3):
                masked_image[:, :, c] = im[:, :, c] * poubelle_mask

            masked_image[:, :, 3] = poubelle_mask.astype(np.uint8) * 255
            filename = os.path.basename(image_path)
            if online:
                filename = filename+f"_mask_{i}.png"

            elif ".jpeg" in image_path:
                filename = filename.replace(".jpeg", f"_mask_{i}.png")

            output_path = os.path.join(output_dir, filename)
            cv2.imwrite(output_path, masked_image)
            print(f"✅ Image sauvegardée : {output_path}")
            saved_files.append(output_path)

    return saved_files

# ===== Fonctions pour optimiser et déplacer =====

def calculate_iou(mask, box_points):
    contour = np.int32(box_points)
    mask_box = np.zeros_like(mask)
    cv2.fillPoly(mask_box, [contour], 255)
    intersection = np.logical_and(mask > 0, mask_box > 0)
    union = np.logical_or(mask > 0, mask_box > 0)
    iou = np.sum(intersection) / np.sum(union) if np.sum(union) != 0 else 0
    return iou

def cost_function(params, mask):
    box_points = np.array(params).reshape((4, 2))
    iou = calculate_iou(mask, box_points)
    return 1 - iou

def optimize_box(mask, initial_box):
    initial_params = initial_box.flatten()
    result = minimize(
        cost_function,
        initial_params,
        args=(mask,),
        method='Powell',
        options={'maxiter': 2000, 'disp': False}
    )
    optimized_box = result.x.reshape((4, 2))
    return optimized_box

def find_lowest_point(mask):
    """Trouver le point le plus bas dans un mask binaire."""
    ys, xs = np.where(mask > 0)
    if len(ys) == 0:
        return None
    idx = np.argmax(ys)
    return np.array([xs[idx], ys[idx]])

# def move_trapezoid(trap, target_point):
#     """Déplacer tout le trapèze pour aligner son sommet bas avec target_point."""
#     current_lowest_idx = np.argmax(trap[:, 1])  # index sommet le plus bas
#     current_lowest_point = trap[current_lowest_idx]
#     translation_vector = target_point - current_lowest_point
#     translation_vector[0] = 0  # Déplacer seulement en hauteur
#     new_trap = trap + translation_vector
#     return new_trap
def calculer_centre_trapeze(points):
    """
    Calcule le centre d'un trapèze défini par 4 points.
    Le centre est défini comme le point moyen des 4 sommets.
    
    Args:
        points: Liste de 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
    
    Returns:
        (centre_x, centre_y): Coordonnées du centre
    """
    # Convertir en array numpy pour faciliter les calculs
    points_array = np.array(points)
    
    # Calculer le centre comme la moyenne des coordonnées des points
    centre = np.mean(points_array, axis=0)
    
    return centre[0], centre[1]

def reduire_trapeze(points, centre, rapport):
    """
    Crée un nouveau trapèze réduit avec le même centre et un rapport de réduction.
    
    Args:
        points: Liste de 4 points [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        centre: (centre_x, centre_y) - Coordonnées du centre du trapèze
        rapport: Facteur de réduction (0 < rapport < 1 pour réduire)
    
    Returns:
        Liste des points du nouveau trapèze
    """
    # Convertir en array numpy
    points_array = np.array(points)
    centre_array = np.array(centre)
    
    # Appliquer la réduction homothétique
    # Pour chaque point P, le nouveau point P' est:
    # P' = centre + rapport * (P - centre)
    nouveau_trapeze = centre_array + rapport * (points_array - centre_array)
    
    return nouveau_trapeze.tolist()

def move_trapezoid(trap, target_point,isParoiOutExist):
    if isParoiOutExist:
        trap_copy = trap[trap[:, 0].argsort()]
        print(trap_copy[0:1])

        current_lowest_idx = np.argmax(trap_copy[0:2][:,1])  # index sommet le plus bas
        current_lowest_point = trap_copy[current_lowest_idx]
        print(target_point, current_lowest_point,"ichigo")
        translation_vector = target_point - current_lowest_point
        translation_vector[0]=0
        new_trap = trap + translation_vector
    else:
        centre = calculer_centre_trapeze(trap)
        rapport_reduction = 0.5  # Réduction de 50%
        new_trap = np.array(reduire_trapeze(trap, centre, rapport_reduction))

    return new_trap

def move_trapezoid_strat2(trap, paroi_mask, isParoiOutExist, image,im,target_point):
    cv2.circle(im, target_point, radius=10, color=(0, 255, 255), thickness=-1)
    def est_en_contact(sommet, paroi_mask, rayon):
        mask_contact = np.zeros_like(paroi_mask, dtype=np.uint8)
        cv2.circle(mask_contact, tuple(sommet), rayon, 255, -1)
        return cv2.bitwise_and(paroi_mask, mask_contact).any()

    # def point_le_plus_bas(paroi_mask):
    #     points = np.column_stack(np.where(paroi_mask > 0))
    #     if points.size == 0:
    #         return None
    #     return points[np.argmax(points[:, 0])][::-1]  # Retourne (x, y)


    def point_le_plus_bas(paroi_mask):
        """
        Détecte les coins dans le masque binaire, et retourne celui le plus bas (coordonnée y maximale).
        """
        # Convertir le masque en format compatible pour la détection
        mask_float = paroi_mask.astype(np.uint8)
        if mask_float.max() <= 1:
            mask_float *= 255  # S'assurer que les pixels sont bien à 255

        # Détection des coins
        corners = cv2.goodFeaturesToTrack(mask_float, maxCorners=100, qualityLevel=0.01, minDistance=10)

        if corners is None:
            return None

        # Trouver le coin avec la plus grande valeur en y (le plus bas)
        corners = [tuple(map(int, corner.ravel())) for corner in corners]
        coin_plus_bas = max(corners, key=lambda pt: pt[1])  # pt[1] = y

        return coin_plus_bas  # (x, y)


    def point_haut_droite(trap):
        trap = np.array(trap)
        y_min = np.min(trap[:, 1])
        candidats = trap[trap[:, 1] == y_min]
        point = candidats[np.argmax(candidats[:, 0])]
        return point

    def trouver_point_regression(paroi_mask, pt_hd, pt_bas, r2,im):
        contours, _ = cv2.findContours(paroi_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if not contours:
            return None

        contour = max(contours, key=cv2.contourArea)  # On suppose qu'il y a une seule paroi
        min_dist_hd = float('inf')
        pt_hd_near = None
        index_hd_near = None
        min_dist_bas = float('inf')
        pt_bas_near = None
        index_bas_near = None



        # Parcourir tous les points du plus grand contour
        for index_pt, pt in enumerate(contour):
            pt_contour = tuple(pt[0])  # Chaque point du contour est dans le format (x, y)
            
            # Calculer la distance entre pt_hd et pt_contour
            dist_hd = np.linalg.norm(np.array(pt_contour) - np.array(pt_hd))
            dist_bas = np.linalg.norm(np.array(pt_contour) - np.array(pt_bas))
            
            if dist_hd < min_dist_hd:
                min_dist_hd = dist_hd
                pt_hd_near = pt_contour
                index_hd_near = index_pt

            if dist_bas < min_dist_bas:
                min_dist_bas = dist_bas
                pt_bas_near = pt_contour
                index_bas_near = index_pt

        print(index_bas_near,index_hd_near,"itachi2")
        sous_contour = contour[index_hd_near:index_bas_near:-1]
        sous_contour = sous_contour.reshape(-1, 2)

        # sous_contour = [pt for pt in points if y_min <= pt[1] <= y_max and x_max>= pt[0] >= x_min]
        sous_contour_np = np.array(sous_contour, dtype=np.int32).reshape((-1, 1, 2))

        cv2.polylines(im, [sous_contour_np], isClosed=False, color=(52, 186, 235), thickness=10)

        # Trier les points du haut vers le bas
        #sous_contour = sorted(sous_contour, key=lambda pt: pt[1])
        #sous_contour.sort(key=lambda pt: pt[1])

        for i in range(len(sous_contour),5,-1):
            pts_reg = sous_contour[:i]
            X = np.array([pt[0] for pt in pts_reg]).reshape(-1, 1)
            Y = np.array([pt[1] for pt in pts_reg])
            reg = LinearRegression().fit(X, Y)
            m = reg.score(X, Y)
            print("hello bru",m)
            if m > r2:
                # x_val = np.array([[pt_bas[0]]])
                # y_pred = reg.predict(x_val)
                x_val = np.array([[pt_bas[0]]])
                y_pred = reg.predict(x_val)
                a = reg.coef_[0]
                b = reg.intercept_
                print(f"Prédiction: x={x_val[0][0]}, y={y_pred[0]},a={a}")

                # Tracer la droite de régression sur l'image
                # Calculer les extrémités de la droite de régression
                x1, x2 = 1024,pt_bas[0]
                y1, y2 = reg.predict([[x1]]), reg.predict([[x2]])

                # Tracer la ligne de régression sur l'image
                start_point = (int(x1), int(y1))
                end_point = (int(x2), int(y2))

                # Tracer la droite de régression en vert
                cv2.line(im, start_point, end_point, (0, 255, 0), 20)

                return (int( (pt_bas[1]- b) / a),int(pt_bas[1])),a

        return None,None

    rayon_contact = 10  # Rayon utilisé pour vérifier le contact

    trap = np.array(trap, dtype=np.int32)

    if isParoiOutExist:
        print("la paroi existe")
        sommet_init = point_haut_droite(trap)
        index_current = np.where((trap == sommet_init).all(axis=1))[0][0]

        new_trap = trap.copy()  # Par défaut, pas de déplacement
        point_bas = trap[0]
        sommet = trap[0]
        for _ in range(4):
            sommet = trap[index_current]
            if est_en_contact(sommet, paroi_mask, rayon_contact):
                point_bas = point_le_plus_bas(paroi_mask)
                if point_bas is not None:
                    print("ichi",point_bas)
                    xy,coeff_dir = trouver_point_regression(paroi_mask, sommet, point_bas, 0.8,im)
                    point_bas = xy if xy else point_bas
                    print("ni",point_bas)
                    vecteur = np.array(point_bas) - np.array(sommet)
                    if coeff_dir:
                        new_trap = [tuple(np.array(p) + vecteur) for p in trap] if xy and coeff_dir>0.5 else move_trapezoid(trap, target_point,isParoiOutExist)
                    else:
                        new_trap = move_trapezoid(trap, target_point,isParoiOutExist)

                break
            index_current = (index_current + 1) % 4
    else:
        centre = calculer_centre_trapeze(trap)
        rapport_reduction = 0.5
        new_trap = np.array(reduire_trapeze(trap, centre, rapport_reduction))

    return new_trap #,tuple(sommet),tuple(point_bas)

def order_trapezoid_points(points, suffix):
    """Ordonne les points d'un trapèze et ajoute un suffix (_t ou _b)."""
    sorted_by_y = points[np.argsort(points[:, 1])]
    bottom_two = sorted_by_y[:2]
    top_two = sorted_by_y[2:]

    bottom_two = bottom_two[np.argsort(bottom_two[:, 0])]
    top_two = top_two[np.argsort(top_two[:, 0])]

    ordered = np.vstack([bottom_two, top_two[::-1]])  # x1, x2, x3, x4

    names = [f"x1{suffix}", f"x2{suffix}", f"x3{suffix}", f"x4{suffix}"]
    points_list = [{"name": name, "x": float(pt[0]), "y": float(pt[1])} for name, pt in zip(names, ordered)]

    return points_list

# ===== Fonction principale : destruct_cropped_img =====

def destruct_cropped_img(image_path,output_dir="app/output_cropped",online=False):
    predictor = PredictorSingleton.get_instance()

    if online:
        im = download_image(image_path)
    else:
        im = cv2.imread(image_path)

    # Prédictions
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")

    # Récupérer la metadata
    metadata = MetadataCatalog.get("my_dataset_train")
    class_names = metadata.get("thing_classes", None)

    if class_names is None:
        raise ValueError("thing_classes non trouvé dans MetadataCatalog")

    masks = instances.pred_masks.numpy()
    pred_classes = instances.pred_classes.numpy()

    # =========== Traitement pour chaque "poubelle" =============

    all_outputs = []

    poubelle_indices = [i for i, cls in enumerate(pred_classes) if class_names[cls] == "poubelle"]
    top_indices = [i for i, cls in enumerate(pred_classes) if class_names[cls] == "top"]
    paroi_out_indices = [i for i, cls in enumerate(pred_classes) if class_names[cls] == "paroi_out"]

    if not poubelle_indices:
        print("Aucune 'poubelle' trouvée.")
    else:
        im_visu = im.copy()
        for pid in poubelle_indices:
            mask_poubelle = (masks[pid] * 255).astype(np.uint8)

            idx_paroi_out = None

            for id_o in paroi_out_indices:
                mask_o = (masks[id_o] * 255).astype(np.uint8)
                overlap_o = np.sum(np.logical_and(mask_o > 0, mask_poubelle > 0))
                if overlap_o > 0:
                    idx_paroi_out = id_o
                    break

            for tid in top_indices:
                mask_top = (masks[tid] * 255).astype(np.uint8)

                # Vérifier que le top est dans la poubelle
                overlap = np.sum(np.logical_and(mask_top > 0, mask_poubelle > 0))
                if overlap == 0:
                    continue

                # ===== Optimiser trapèze rouge (haut) =====
                contours, _ = cv2.findContours(mask_top, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                rect = cv2.minAreaRect(contours[0])
                box = cv2.boxPoints(rect)
                box = np.array(box)

                optimized_box = optimize_box(mask_top, box)
                lowest_point = find_lowest_point(mask_poubelle)
                #moved_trap = move_trapezoid(optimized_box, lowest_point,idx_paroi_out)
                h,w = im.shape[:2]
                image_bg = np.zeros((h, w), dtype=np.uint8)
                print(idx_paroi_out,"danzo")
                if isinstance(idx_paroi_out, int):
                    moved_trap = move_trapezoid_strat2(optimized_box, mask_o, isinstance(idx_paroi_out, int), image_bg,im_visu,lowest_point)
                else:
                    moved_trap = move_trapezoid(optimized_box, lowest_point,idx_paroi_out)

                # # ===== Déplacer tout le trapèze vers le point bas de la poubelle =====
                # lowest_point = find_lowest_point(mask_poubelle)
                # if lowest_point is not None:
                #     moved_trap = move_trapezoid(optimized_box, lowest_point)
                # else:
                #     moved_trap = optimized_box.copy()

                # 📦 Générer le dict
                points2D_top = order_trapezoid_points(np.array(optimized_box), "_t")
                points2D_bottom = order_trapezoid_points(np.array(moved_trap), "_b")

                all_points2D = points2D_top + points2D_bottom

                output_dict = {
                    "method": "optimizerTrap",
                    "points2D": all_points2D,
                    "img": image_path,
                    "point3D": None,
                    "param_opti": None,
                    "iou": None
                }

                all_outputs.append(output_dict)
                    # === Dessiner les trapèzes sur l'image originale ===
                

                # Bleu : trapèze haut (optimisé)
                optimized_box_int = np.array(optimized_box, dtype=np.int32)
                cv2.polylines(im_visu, [optimized_box_int], isClosed=True, color=(255, 0, 0), thickness=2)

                # Rouge : trapèze bas (déplacé)
                moved_trap_int = np.array(moved_trap, dtype=np.int32)
                cv2.polylines(im_visu, [moved_trap_int], isClosed=True, color=(0, 0, 255), thickness=2)

                # cv2.circle(im_visu, summit, radius=5, color=(0, 0, 255), thickness=-1)
                
                # cv2.circle(im_visu, point_bas, radius=5, color=(255, 255, 255), thickness=-1)
                # if mask_o:
                #     contours, _ = cv2.findContours(mask_o, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                cv2.drawContours(im_visu, contours, -1, (255, 0, 255), 2)
                # Sauvegarder dans le dossier output_dir
    os.makedirs("test972", exist_ok=True)
    out_filename = os.path.join(os.path.join("test972"), f"visu_{os.path.basename(image_path)}")
    cv2.imwrite(out_filename, im_visu)


    return all_outputs


def calculate_fill_rate(image_path, score_thresh=0.6):
    predictor = PredictorSingleton.get_instance()
    im = cv2.imread(image_path)
    outputs = predictor(im)
    instances = outputs["instances"].to("cpu")

    boxes = instances.pred_boxes
    scores = instances.scores
    classes = instances.pred_classes
    masks = instances.pred_masks
    poubelle_class_id = MetadataCatalog.get("my_dataset_train").thing_classes.index("poubelle")
    paroi_int_class_id = MetadataCatalog.get("my_dataset_train").thing_classes.index("paroi_int")
    paroi_out_class_id = MetadataCatalog.get("my_dataset_train").thing_classes.index("paroi_out")

    for i in range(len(classes)):
        if classes[i] == poubelle_class_id and scores[i] >= score_thresh:
            poubelle_mask = masks[i].numpy()

            # Vérifier si un autre masque est entièrement contenu
            for j in range(len(classes)):
                if i != j and masks[j].numpy().sum() > 0:
                    intersection = np.logical_and(poubelle_mask, masks[j].numpy())
                    if np.all(intersection == masks[j].numpy()):
                        pass

            masked_image = np.zeros((im.shape[0], im.shape[1], 4), dtype=np.uint8)

            for c in range(3):
                masked_image[:, :, c] = im[:, :, c] * poubelle_mask

            masked_image[:, :, 3] = poubelle_mask.astype(np.uint8) * 255

            # Initialisation des valeurs de paroi_int et paroi_out
            paroi_int = None
            paroi_out = None

            # Parcours des instances pour récupérer paroi_int et paroi_out
            for j in range(len(classes)):
                if classes[j] == paroi_int_class_id and scores[j] >= score_thresh:
                    paroi_int = masks[j].numpy()
                elif classes[j] == paroi_out_class_id and scores[j] >= score_thresh:
                    paroi_out = masks[j].numpy()

            # Calcul de full_bin
            if paroi_int is None:
                full_bin = 1
            else:
                if paroi_out is None or np.sum(paroi_out) < np.sum(paroi_int):
                    paroi_out = 2 * np.sum(paroi_int)
                full_bin = 1 - np.sum(paroi_int) / np.sum(paroi_out)

    return {"full_bin": full_bin,"paroi_int":int(np.sum(paroi_int)),"paroi_out":int(np.sum(paroi_out))}

