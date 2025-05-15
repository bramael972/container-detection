import numpy as np
import matplotlib.pyplot as plt
import cv2
from PIL import Image
from scipy.optimize import differential_evolution, minimize
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

        # Projection orthographique
        # P = np.array([
        #     [1, 0, 0],
        #     [0, 1, 0]
        # ])
P = np.array([
            [1, 0, 0],  # on garde Y
            [0, 1, 0]   # on garde Z
        ])
R_flip = np.array([
    [-1, 0, 0],
    [ 0, -1, 0],
    [ 0, 0, 1]
])
# --- Fonction pour calculer l'IoU (Intersection over Union) ---
def calculate_iou(mask1, mask2):
    intersection = np.logical_and(mask1, mask2).sum()
    union = np.logical_or(mask1, mask2).sum()
    if union == 0:
        return 0
    return intersection / union

# --- Fonction à optimiser ---
def evaluate_params(params, target_mask, h, w, cx_target, cy_target):
    # Extraire les paramètres
    l, L, scale, theta_deg, phi_deg, dx, dy = params

    # Limiter les paramètres à des valeurs raisonnables
    l = max(0.5, min(10, l))
    L = max(0.5, min(20, L))
    scale = max(10, min(300, scale))

    # Créer le pavé
    cube_vertices = np.array([
        [0, 0, 0], [l, 0, 0], [l, l, 0], [0, l, 0],
        [0, 0, L], [l, 0, L], [l, l, L], [0, l, L]
    ]) * scale

    # Angles de caméra
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)

    # Rotations
    # Ry = np.array([
    #     [np.cos(theta), 0, np.sin(theta)],
    #     [0, 1, 0],
    #     [-np.sin(theta), 0, np.cos(theta)]
    # ])
    # Rx = np.array([
    #     [1, 0, 0],
    #     [0, np.cos(phi), -np.sin(phi)],
    #     [0, np.sin(phi), np.cos(phi)]
    # ])
    # rotation_matrix = Rx @ Ry

    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    rotation_matrix = Ry @ Rz

    rotated_vertices = (rotation_matrix @ cube_vertices.T).T

    # Projection orthographique
    # P = np.array([
    #     [1, 0, 0],
    #     [0, 1, 0]
    # ])
    # P = np.array([
    #     [0, 1, 0],  # on garde Y
    #     [0, 0, 1]   # on garde Z
    # ])
    rotated_vertices = (R_flip @ rotated_vertices.T).T
    projected = (P @ rotated_vertices.T).T

    # Translation personnalisée (vers le centre de gravité + décalage)
    center_proj = projected.mean(axis=0)
    translation = np.array([cx_target + dx, cy_target + dy]) - center_proj
    projected_translated = projected + translation

    # Création du masque du pavé
    mask_pave = np.zeros((h, w), dtype=np.uint8)
    pts = projected_translated.astype(np.int32)

    # Vérification que les points sont valides pour convexHull
    if len(pts) >= 3:
        try:
            hull = cv2.convexHull(pts)
            if len(hull) >= 3:  # S'assurer que hull a au moins 3 points
                cv2.fillPoly(mask_pave, [hull], 255)
        except:
            # Si convexHull échoue, retourner un mauvais score
            return 0.0

    # Calcul de l'IoU
    mask_pave_bool = mask_pave > 0
    target_mask_bool = target_mask > 0
    iou = calculate_iou(mask_pave_bool, target_mask_bool)

    # On veut maximiser l'IoU, donc on retourne -iou (minimize cherche un minimum)
    return -iou

# --- Fonction de callback pour suivre la progression ---
def callback_progress(xk, convergence=0):
    global global_target_mask, global_h, global_w, global_cx, global_cy
    print(f"Progression : IoU actuel = {-evaluate_params(xk, global_target_mask, global_h, global_w, global_cx, global_cy):.4f}")
    return False

# --- Fonction principale d'optimisation ---
def optimize_cube_parameters(img_path, start_params=None, refine=False):
    # Charger l'image avec transparence
    img = Image.open(img_path).convert("RGBA")
    img_np = np.array(img)
    h, w = img_np.shape[:2]
    alpha = img_np[:, :, 3]
    target_mask = (alpha > 0).astype(np.uint8) * 255

    # Centre de gravité du masque
    M = cv2.moments(target_mask)
    if M["m00"] != 0:
        cx_target = int(M["m10"] / M["m00"])
        cy_target = int(M["m01"] / M["m00"])
    else:
        cx_target, cy_target = w // 2, h // 2

    # Si on est en mode raffinement
    if refine and start_params is not None:
        # Bornes resserrées autour des meilleurs paramètres précédents
        l_opt, L_opt, scale_opt, theta_opt, phi_opt, dx_opt, dy_opt = start_params
        bounds = [
            (max(0.5, l_opt * 0.8), min(10, l_opt * 1.2)),          # l ±20%
            (max(0.5, L_opt * 0.8), min(20, L_opt * 1.2)),          # L ±20%
            (max(50, scale_opt * 0.9), min(300, scale_opt * 1.1)),  # scale ±10%
            (max(-45, theta_opt - 20), min(45, theta_opt + 20)),  # theta limité
            (max(-45, phi_opt - 10), min(45, phi_opt + 10)),  # phi limité
            #(theta_opt - 20, theta_opt + 20),                        # theta ±20°
            #(max(-90, phi_opt - 10), min(90, phi_opt + 10)),         # phi ±10°
            (dx_opt - 10, dx_opt + 10),                              # dx ±10 pixels
            (dy_opt - 10, dy_opt + 10)                               # dy ±10 pixels
        ]

        # Population et itérations augmentées pour la phase de raffinement
        popsize = 20
        maxiter = 50
    else:
        # Bornes originales pour la recherche globale
        bounds = [(0.5, 10),     # l
                  (0.5, 20),     # L
                  (50, 300),     # scale
                    (-45, 45),     # theta_deg (limité)
                    (-45, 45),     # phi_deg (limité)
                #   (0, 360),      # theta_deg
                #   (-90, 90),     # phi_deg
                  (-150, 150),   # dx
                  (-150, 150)]   # dy

        # Paramètres standards
        popsize = 15
        maxiter = 30

    # Optimisation avec algorithme évolutif
    print(f"{'Raffinement' if refine else 'Démarrage'} de l'optimisation...")

    result = differential_evolution(
        evaluate_params,
        bounds,
        args=(target_mask, h, w, cx_target, cy_target),
        popsize=popsize,
        maxiter=maxiter,  # Nombre de générations
        tol=0.001,  # Tolérance plus stricte
        polish=True,  # Application d'une optimisation locale à la fin
        init='latinhypercube',  # Méthode d'initialisation plus diversifiée
        mutation=(0.5, 1.0),  # Stratégie de mutation adaptative
        recombination=0.8,  # Augmentation du taux de recombinaison
        callback=callback_progress
    )

    # Extraction des paramètres optimaux
    l_opt, L_opt, scale_opt, theta_opt, phi_opt, dx_opt, dy_opt = result.x
    print(f"Paramètres optimaux trouvés: l={l_opt:.2f}, L={L_opt:.2f}, scale={scale_opt:.2f}, "
          f"theta={theta_opt:.2f}°, phi={phi_opt:.2f}°, dx={dx_opt:.2f}, dy={dy_opt:.2f}")
    print(f"IoU optimal: {-result.fun:.4f}")

    return result.x, -result.fun

def visualize_result(img_np, target_mask, l, L, scale, theta_deg, phi_deg,
                     cx_target, cy_target, dx, dy, h, w):
    # Masque de l'objet en bleu
    img_bg = img_np[:, :, :3].copy()
    blue = np.array([0, 0, 255], dtype=np.uint8)
    target_mask_bool = target_mask > 0
    img_bg[target_mask_bool] = blue

    # Création du pavé
    cube_vertices = np.array([ 
        [0, 0, 0], [l, 0, 0], [l, l, 0], [0, l, 0], 
        [0, 0, L], [l, 0, L], [l, l, L], [0, l, L] 
    ]) * scale

    # Angles de caméra
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)

    # Rotations
    # Ry = np.array([ 
    #     [np.cos(theta), 0, np.sin(theta)], 
    #     [0, 1, 0], 
    #     [-np.sin(theta), 0, np.cos(theta)] 
    # ])
    # Rx = np.array([ 
    #     [1, 0, 0], 
    #     [0, np.cos(phi), -np.sin(phi)], 
    #     [0, np.sin(phi), np.cos(phi)] 
    # ])
    # rotation_matrix = Rx @ Ry
    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    rotation_matrix = Ry @ Rz
    rotated_vertices = (rotation_matrix @ cube_vertices.T).T

    # Projection orthographique
    # P = np.array([ 
    #     [1, 0, 0], 
    #     [0, 1, 0] 
    # ])
    # P = np.array([
    #     [0, 1, 0],  # on garde Y
    #     [0, 0, 1]   # on garde Z
    # ])
    rotated_vertices = (R_flip @ rotated_vertices.T).T
    projected = (P @ rotated_vertices.T).T


    # Translation vers le centre de gravité de l'objet + décalage optimal
    center_proj = projected.mean(axis=0)
    translation = np.array([cx_target + dx, cy_target + dy]) - center_proj
    projected_translated = projected + translation

    # Création du masque du trapézoïde
    mask_trapezoid = np.zeros((h, w), dtype=np.uint8)
    pts = projected_translated.astype(np.int32)

    try:
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask_trapezoid, [hull], 255)
    except:
        print("Erreur lors de la création du hull convexe")

    # Appliquer le trapézoïde en vert sur l'image
    img_result = img_bg.copy()
    mask_trapezoid_bool = mask_trapezoid > 0
    img_result[mask_trapezoid_bool] = [0, 255, 0]  # vert pour le trapézoïde

    # Calculer l'intersection et visualiser en rouge
    intersection = np.logical_and(target_mask_bool, mask_trapezoid_bool)
    img_result[intersection] = [255, 0, 0]  # rouge pour l'intersection

    # Calcul de l'IoU final
    iou = calculate_iou(target_mask_bool, mask_trapezoid_bool)

    # Affichage final avec les deux masques et leur intersection
    plt.figure(figsize=(12, 8))

    # Image avec les deux masques
    plt.subplot(1, 2, 1)
    plt.imshow(img_result)

    # NOUVEAU: Dessiner les arêtes du trapézoïde projeté sur l'image des masques
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),  # Base inférieure
        (4, 5), (5, 6), (6, 7), (7, 4),  # Base supérieure
        (0, 4), (1, 5), (2, 6), (3, 7)   # Arêtes verticales
    ]

    for i, j in edges:
        plt.plot([projected_translated[i, 0], projected_translated[j, 0]],
                 [projected_translated[i, 1], projected_translated[j, 1]],
                 'y-', linewidth=2)  # Arêtes en jaune pour bien les voir

    plt.title(f"Masques superposés (IoU = {iou:.4f})")
    plt.axis('off')

    # Légende
    legend_img = np.zeros((100, 300, 3), dtype=np.uint8)
    legend_img[20:40, 20:100] = [0, 0, 255]  # bleu - objet original
    legend_img[50:70, 20:100] = [0, 255, 0]  # vert - trapézoïde
    legend_img[80:100, 20:100] = [255, 0, 0]  # rouge - intersection

    plt.subplot(1, 2, 2)
    plt.imshow(legend_img)
    plt.text(110, 30, "Masque de l'objet", fontsize=10, color='white')
    plt.text(110, 60, "Masque du trapézoïde", fontsize=10, color='white')
    plt.text(110, 90, "Intersection", fontsize=10, color='white')

    # NOUVEAU: Ajouter une ligne pour la légende des arêtes
    plt.plot([20, 100], [110, 110], 'y-', linewidth=2)
    plt.text(110, 110, "Arêtes du trapézoïde", fontsize=10, color='white')

    plt.axis('off')
    plt.title("Légende")

    plt.tight_layout()
    plt.savefig("app/output_keypoints/resultat_optimisation_trapezoid.png")
    #plt.show()

    # Afficher les arêtes du trapézoïde en 3D
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')

    # Dessiner les arêtes du trapézoïde
    for i, j in edges:
        ax.plot([cube_vertices[i, 0], cube_vertices[j, 0]],
                [cube_vertices[i, 1], cube_vertices[j, 1]],
                [cube_vertices[i, 2], cube_vertices[j, 2]], 'k-')

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.set_title(f"Trapézoïde 3D optimal")

    # Centrer la vue
    max_range = np.array([cube_vertices[:, 0].max() - cube_vertices[:, 0].min(),
                          cube_vertices[:, 1].max() - cube_vertices[:, 1].min(),
                          cube_vertices[:, 2].max() - cube_vertices[:, 2].min()]).max() / 2.0
    mid_x = (cube_vertices[:, 0].max() + cube_vertices[:, 0].min()) * 0.5
    mid_y = (cube_vertices[:, 1].max() + cube_vertices[:, 1].min()) * 0.5
    mid_z = (cube_vertices[:, 2].max() + cube_vertices[:, 2].min()) * 0.5

    ax.set_xlim(mid_x - max_range, mid_x + max_range)
    ax.set_ylim(mid_y - max_range, mid_y + max_range)
    ax.set_zlim(mid_z - max_range, mid_z + max_range)

    plt.savefig("app/output_keypoints/trapezoid_3d.png")
    #plt.show()

    return iou


def get_dict_result(img_np, target_mask, l, L, scale, theta_deg, phi_deg,
                    cx_target, cy_target, dx, dy, h, w, img_path):
    # Masque de l'objet en bleu
    img_bg = img_np[:, :, :3].copy()
    blue = np.array([0, 0, 255], dtype=np.uint8)
    target_mask_bool = target_mask > 0
    img_bg[target_mask_bool] = blue

    # Création du pavé
    cube_vertices = np.array([ 
        [0, 0, 0], [l, 0, 0], [l, l, 0], [0, l, 0], 
        [0, 0, L], [l, 0, L], [l, l, L], [0, l, L] 
    ]) * scale

    # Angles de caméra
    theta, phi = np.radians(theta_deg), np.radians(phi_deg)

    # Rotations
    # Ry = np.array([ 
    #     [np.cos(theta), 0, np.sin(theta)], 
    #     [0, 1, 0], 
    #     [-np.sin(theta), 0, np.cos(theta)] 
    # ])
    # Rx = np.array([ 
    #     [1, 0, 0], 
    #     [0, np.cos(phi), -np.sin(phi)], 
    #     [0, np.sin(phi), np.cos(phi)] 
    # ])
    # rotation_matrix = Rx @ Ry

    Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0, 0, 1]
    ])

    Ry = np.array([
        [np.cos(theta), 0, np.sin(theta)],
        [0, 1, 0],
        [-np.sin(theta), 0, np.cos(theta)]
    ])

    rotation_matrix = Ry @ Rz


    rotated_vertices = (rotation_matrix @ cube_vertices.T).T  # (8, 3)

    # Projection orthographique
    # P = np.array([ 
    #     [1, 0, 0], 
    #     [0, 1, 0] 
    # ])
    # P = np.array([
    #     [0, 1, 0],  # on garde Y
    #     [0, 0, 1]   # on garde Z
    # ])
    rotated_vertices = (R_flip @ rotated_vertices.T).T
    projected = (P @ rotated_vertices.T).T  # (8, 2)

    # Translation vers le centre de gravité de l'objet + décalage optimal
    center_proj = projected.mean(axis=0)
    translation = np.array([cx_target + dx, cy_target + dy]) - center_proj
    projected_translated = projected + translation  # (8, 2)

    # Création du masque du trapézoïde
    mask_trapezoid = np.zeros((h, w), dtype=np.uint8)
    pts = projected_translated.astype(np.int32)

    try:
        hull = cv2.convexHull(pts)
        cv2.fillPoly(mask_trapezoid, [hull], 255)
    except:
        print("Erreur lors de la création du hull convexe")

    mask_trapezoid_bool = mask_trapezoid > 0

    # Calcul de l'IoU final
    iou = calculate_iou(target_mask_bool, mask_trapezoid_bool)

    # Liste ordonnée des noms de points
    keypoint_names = [
        "x1_b", "x2_b", "x3_b", "x4_b",
        "x1_t", "x2_t", "x3_t", "x4_t"
    ]

    # Construction du dict points3D
    point3D = []
    for i, name in enumerate(keypoint_names):
        x, y, z = rotated_vertices[i]
        point3D.append({
            "name": name,
            "x": float(x),
            "y": float(y),
            "z": float(z)
        })

    # Construction du dict points2D
    points2D = []
    for i, name in enumerate(keypoint_names):
        x, y = projected_translated[i]
        points2D.append({
            "name": name,
            "x": float(x),
            "y": float(y),
            "score": None
        })

    # Paramètres d’optimisation utilisés
    param_opti = {
        "l": l,
        "L": L,
        "theta": theta_deg,
        "phi": phi_deg,
        "scale": scale
    }

    return {
        "method": "optimizerKeyPoint",
        "points2D": points2D,
        "img": img_path,
        "point3D": point3D,
        "param_opti": param_opti,
        "iou": iou
    }



# --- Fonction pour tester une solution précise avec de petites variations ---
def fine_tune_around_solution(params, target_mask, h, w, cx_target, cy_target, num_tries=1000):
    best_params = params.copy()
    best_iou = -evaluate_params(params, target_mask, h, w, cx_target, cy_target)

    print(f"Début du fine-tuning. IoU initial: {best_iou:.4f}")

    improvement_count = 0
    for i in range(num_tries):
        # Créer une variation des paramètres actuels
        trial_params = best_params.copy()

        # Ajuster avec des variations plus petites à mesure qu'on progresse
        factor = max(0.1, 1.0 - (i / num_tries) * 0.9)  # Diminue progressivement

        # Variations aléatoires avec diminution progressive
        trial_params[0] *= np.random.uniform(1 - 0.05 * factor, 1 + 0.05 * factor)  # l
        trial_params[1] *= np.random.uniform(1 - 0.05 * factor, 1 + 0.05 * factor)  # L
        trial_params[2] *= np.random.uniform(1 - 0.03 * factor, 1 + 0.03 * factor)  # scale
        trial_params[3] += np.random.uniform(-5 * factor, 5 * factor)               # theta
        trial_params[4] += np.random.uniform(-3 * factor, 3 * factor)               # phi
        trial_params[5] += np.random.uniform(-3 * factor, 3 * factor)               # dx
        trial_params[6] += np.random.uniform(-3 * factor, 3 * factor)               # dy

        # Vérifier l'IoU
        trial_iou = -evaluate_params(trial_params, target_mask, h, w, cx_target, cy_target)

        # Mise à jour si amélioration
        if trial_iou > best_iou:
            best_iou = trial_iou
            best_params = trial_params.copy()
            improvement_count += 1
            print(f"Amélioration #{improvement_count}: IoU = {best_iou:.4f}")

            # Si on dépasse 0.9, on peut s'arrêter
            if best_iou > 0.9:
                print("Objectif IoU > 0.9 atteint!")
                break

        # Afficher la progression
        if (i+1) % 100 == 0:
            print(f"Progression: {i+1}/{num_tries}, Meilleur IoU: {best_iou:.4f}")

    return best_params, best_iou

# --- Fonction pour fusion des masques pour améliorer l'IoU ---
def optimize_mask_combination(params, target_mask, h, w, cx_target, cy_target):
    """Tente de fusionner plusieurs masques de parallélépipèdes avec des orientations légèrement différentes"""

    # Créer trois versions légèrement différentes du parallélépipède
    base_params = params.copy()

    # Créer les variantes
    variants = [
        base_params.copy(),  # Original
        base_params.copy(),  # Variante 1
        base_params.copy()   # Variante 2
    ]

    # Ajuster légèrement les angles pour les variantes
    variants[1][3] += 5      # Différent theta (+5°)
    variants[1][4] -= 2      # Différent phi (-2°)

    variants[2][3] -= 5      # Différent theta (-5°)
    variants[2][4] += 2      # Différent phi (+2°)

    # Créer un masque combiné
    combined_mask = np.zeros((h, w), dtype=np.uint8)

    # Ajouter chaque variante au masque
    for v_params in variants:
        # Créer le pavé
        l, L, scale, theta_deg, phi_deg, dx, dy = v_params

        # Créer le masque pour cette variante
        cube_vertices = np.array([
            [0, 0, 0], [l, 0, 0], [l, l, 0], [0, l, 0],
            [0, 0, L], [l, 0, L], [l, l, L], [0, l, L]
        ]) * scale

        # Angles de caméra
        theta, phi = np.radians(theta_deg), np.radians(phi_deg)

        # Rotations
        # Ry = np.array([
        #     [np.cos(theta), 0, np.sin(theta)],
        #     [0, 1, 0],
        #     [-np.sin(theta), 0, np.cos(theta)]
        # ])
        # Rx = np.array([
        #     [1, 0, 0],
        #     [0, np.cos(phi), -np.sin(phi)],
        #     [0, np.sin(phi), np.cos(phi)]
        # ])
        # rotation_matrix = Rx @ Ry

        Rz = np.array([
        [np.cos(phi), -np.sin(phi), 0],
        [np.sin(phi),  np.cos(phi), 0],
        [0, 0, 1]
        ])

        Ry = np.array([
            [np.cos(theta), 0, np.sin(theta)],
            [0, 1, 0],
            [-np.sin(theta), 0, np.cos(theta)]
        ])

        rotation_matrix = Ry @ Rz

        rotated_vertices = (rotation_matrix @ cube_vertices.T).T

        # Projection orthographique
        # P = np.array([
        #     [1, 0, 0],
        #     [0, 1, 0]
        # ])
        # P = np.array([
        #     [0, 1, 0],  # on garde Y
        #     [0, 0, 1]   # on garde Z
        # ])
        rotated_vertices = (R_flip @ rotated_vertices.T).T
        projected = (P @ rotated_vertices.T).T

        # Translation personnalisée
        center_proj = projected.mean(axis=0)
        translation = np.array([cx_target + dx, cy_target + dy]) - center_proj
        projected_translated = projected + translation

        # Création du masque pour cette variante
        mask_variant = np.zeros((h, w), dtype=np.uint8)
        pts = projected_translated.astype(np.int32)

        try:
            hull = cv2.convexHull(pts)
            cv2.fillPoly(mask_variant, [hull], 255)

            # Ajouter au masque combiné
            combined_mask = np.logical_or(combined_mask, mask_variant).astype(np.uint8) * 255
        except:
            print("Erreur lors de la création d'une variante du hull")

    # Calculer l'IoU avec le masque combiné
    iou = calculate_iou(target_mask > 0, combined_mask > 0)
    print(f"IoU avec masque combiné: {iou:.4f}")

    # Visualiser le résultat
    img_bg = np.zeros((h, w, 3), dtype=np.uint8)
    img_bg[target_mask > 0] = [0, 0, 255]  # Bleu pour l'objet original
    img_result = img_bg.copy()
    img_result[combined_mask > 0] = [0, 255, 0]  # Vert pour le masque combiné

    # Calculer l'intersection et visualiser en rouge
    intersection = np.logical_and(target_mask > 0, combined_mask > 0)
    img_result[intersection] = [255, 0, 0]  # Rouge pour l'intersection

    plt.figure(figsize=(10, 8))
    plt.imshow(img_result)
    plt.title(f"Masque combiné (IoU = {iou:.4f})")
    plt.axis('off')
    plt.savefig("app/output_keypoints/masque_combine.png")
    #plt.show()

    return iou

# --- Exécution principale ---
def run_optimization_para(img_path):
    #img_path = "app/output_cropped/2023-02-17T16_01_58-camera-ak_e45f0178a01d-R-raw_mask_0.png"  # Chemin vers votre image

    # Variables globales pour le callback
    global global_target_mask, global_h, global_w, global_cx, global_cy

    # Charger l'image et préparer les variables globales
    img = Image.open(img_path).convert("RGBA")
    img_np = np.array(img)
    global_h, global_w = img_np.shape[:2]
    alpha = img_np[:, :, 3]
    global_target_mask = (alpha > 0).astype(np.uint8) * 255

    M = cv2.moments(global_target_mask)
    if M["m00"] != 0:
        global_cx = int(M["m10"] / M["m00"])
        global_cy = int(M["m01"] / M["m00"])
    else:
        global_cx, global_cy = global_w // 2, global_h // 2

    # Phase 1: Optimisation globale
    print("Phase 1: Optimisation globale")
    params_opt1, iou_opt1 = optimize_cube_parameters(img_path)
    print(f"Phase 1 terminée: IoU = {iou_opt1:.4f}")

    # Phase 2: Premier raffinement
    print("\nPhase 2: Premier raffinement")
    params_opt2, iou_opt2 = optimize_cube_parameters(img_path, params_opt1, refine=True)
    print(f"Phase 2 terminée: IoU = {iou_opt2:.4f}")

    # Phase 2.5: Premier raffinement
    print("\nPhase 2.5: Premier raffinement")
    params_opt2, iou_opt2 = optimize_cube_parameters(img_path, params_opt2, refine=True)
    print(f"Phase 2 terminée: IoU = {iou_opt2:.4f}")

    # Phase 3: Second raffinement avec des méthodes d'optimisation locale
    if iou_opt2 < 0.9:  # Seulement si on n'a pas encore atteint l'objectif
        print("\nPhase 3: Second raffinement avec méthodes locales")

        # Essayer avec plusieurs méthodes avancées
        methods = ['Nelder-Mead', 'Powell']
        best_iou = iou_opt2
        best_params = params_opt2

        for method in methods:
            try:
                # Fonction d'enveloppe pour minimize
                def wrapper_func(x):
                    return evaluate_params(x, global_target_mask, global_h, global_w, global_cx, global_cy)

                result = minimize(
                    wrapper_func,
                    params_opt2,
                    method=method,
                    options={'maxiter': 500, 'xatol': 1e-8, 'fatol': 1e-8}
                )

                if -result.fun > best_iou:
                    best_iou = -result.fun
                    best_params = result.x
                    print(f"Amélioration avec {method}: IoU = {best_iou:.4f}")
            except Exception as e:
                print(f"Méthode {method} a échoué: {e}")

        params_opt3, iou_opt3 = best_params, best_iou
        print(f"Phase 3 terminée: IoU = {iou_opt3:.4f}")
    else:
        params_opt3, iou_opt3 = params_opt2, iou_opt2

    # Phase 4: Fine-tuning avec variations aléatoires
    if iou_opt3 < 0.9:
        print("\nPhase 4: Fine-tuning avec variations aléatoires")
        params_opt4, iou_opt4 = fine_tune_around_solution(
            params_opt3, global_target_mask, global_h, global_w, global_cx, global_cy, num_tries=2000
        )
        print(f"Phase 4 terminée: IoU = {iou_opt4:.4f}")
    else:
        params_opt4, iou_opt4 = params_opt3, iou_opt3

    # Phase 5: Essayer une approche de fusion de masques si nécessaire
    if iou_opt4 < 0.9:
        print("\nPhase 5: Essai de fusion de masques")
        iou_combined = optimize_mask_combination(
            params_opt4, global_target_mask, global_h, global_w, global_cx, global_cy
        )
        print(f"Phase 5 terminée: IoU avec masques combinés = {iou_combined:.4f}")

        # Si la fusion améliore significativement
        if iou_combined > iou_opt4 + 0.05:  # Si amélioration d'au moins 5%
            print("La fusion de masques a significativement amélioré l'IoU!")
            final_iou = iou_combined
        else:
            final_iou = iou_opt4
    else:
        final_iou = iou_opt4

    # # Visualisation finale avec les meilleurs paramètres
    # print("\nGénération des résultats finaux...")
    # iou_final = visualize_result(
    #     img_np, global_target_mask,
    #     params_opt4[0], params_opt4[1], params_opt4[2],
    #     params_opt4[3], params_opt4[4],
    #     global_cx, global_cy, params_opt4[5], params_opt4[6],
    #     global_h, global_w
    # )
    return get_dict_result(
        img_np, global_target_mask,
        params_opt4[0], params_opt4[1], params_opt4[2],
        params_opt4[3], params_opt4[4],
        global_cx, global_cy, params_opt4[5], params_opt4[6],
        global_h, global_w,img_path)