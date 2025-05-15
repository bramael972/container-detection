
import cv2
import numpy as np
import itertools
import json

def point_on_mask(point, mask):
    x, y = int(point['x']), int(point['y'])
    h, w = mask.shape
    if 0 <= y < h and 0 <= x < w:
        return mask[y, x] > 0
    return False

def polygon_area(points):
    x = np.array([p['x'] for p in points])
    y = np.array([p['y'] for p in points])
    return 0.5 * np.abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1)))

def shape_similarity(points1, points2):
    # Simple cost based on area difference and point distances
    area1 = polygon_area(points1)
    area2 = polygon_area(points2)
    area_diff = abs(area1 - area2)

    dist_diff = sum(
        np.linalg.norm(
            [points1[i]['x'] - points1[(i + 1) % len(points1)]['x'],
             points1[i]['y'] - points1[(i + 1) % len(points1)]['y']]
        ) -
        np.linalg.norm(
            [points2[i]['x'] - points2[(i + 1) % len(points2)]['x'],
             points2[i]['y'] - points2[(i + 1) % len(points2)]['y']]
        )
        for i in range(len(points1))
    )

    return area_diff + dist_diff  # Cost

def process_trapezoid_info(info, image_path):
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    mask = image[..., 3] if image.shape[2] == 4 else np.ones(image.shape[:2], dtype=np.uint8)

    points = info['points2D']
    top_points = [p for p in points if p['name'].endswith('_t') and p['score'] > 1 and point_on_mask(p, mask)]
    bot_points = [p for p in points if p['name'].endswith('_b') and p['score'] > 1 and point_on_mask(p, mask)]

    best_cost = float('inf')
    best_top = None
    best_bot = None

    for top_comb in itertools.combinations(top_points, min(3, len(top_points))):
        for bot_comb in itertools.combinations(bot_points, min(3, len(bot_points))):
            # To form a trapezoid, complete to 4 by duplicating the last
            top_poly = list(top_comb)
            bot_poly = list(bot_comb)
            if len(top_poly) < 4:
                top_poly += [top_poly[-1]] * (4 - len(top_poly))
            if len(bot_poly) < 4:
                bot_poly += [bot_poly[-1]] * (4 - len(bot_poly))

            cost = shape_similarity(top_poly, bot_poly)
            if cost < best_cost:
                best_cost = cost
                best_top = top_poly
                best_bot = bot_poly

    # Flatten points and maintain their original names
    result_points = []
    if best_top:
        result_points += best_top
    if best_bot:
        result_points += best_bot

    # Remove duplicates (same name)
    unique_result = {}
    for p in result_points:
        unique_result[p['name']] = p
    final_points = list(unique_result.values())

    return {
        "method": "detectronKeyPoint",
        "points2D": final_points,
        "img": info['img'],
        "point3D": None,
        "param_opti": None,
        "iou": None
    }
