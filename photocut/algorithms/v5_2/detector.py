# -*- coding: utf-8 -*-
"""
角落检测器 v4.7 - 基于边缘检测 + ROI 分治
"""

import cv2
import numpy as np
from typing import List, Tuple, Dict, Any, Optional
from photocut.imaging.histogram import ps_levels
from photocut.config import (
    ROI_SCALE,
    PS_LEVELS_HIGHLIGHT,
    HOUGH_RHO_RESOLUTION,
    HOUGH_ANGLE_RESOLUTION,
    SUBPIX_WINDOW_SIZE,
    SUBPIX_CRITERIA_MAX_ITER,
    SUBPIX_CRITERIA_EPSILON,
)

from photocut.detection_parameters import DEFAULT_DETECTION_PARAMETERS, DetectionParameters

def get_roi_regions(width: int, height: int) -> List[Tuple[int, int, int, int]]:
    """
    获取四角 ROI 区域

    返回:
        [(x1, y1, x2, y2), ...] 对应 左上、右上、右下、左下
    """
    roi_w = int(width * ROI_SCALE)
    roi_h = int(height * ROI_SCALE)

    regions = [
        (0, 0, roi_w, roi_h),                    # 左上
        (width - roi_w, 0, width, roi_h),        # 右上
        (width - roi_w, height - roi_h, width, height),  # 右下
        (0, height - roi_h, roi_w, height)       # 左下
    ]

    return regions


def extract_roi(img: np.ndarray, roi: Tuple[int, int, int, int]) -> Tuple[np.ndarray, int, int]:
    """
    提取 ROI 区域

    参数:
        img: 原图
        roi: (x1, y1, x2, y2)

    返回:
        roi_img: ROI 图像
        x_offset: x 偏移量
        y_offset: y 偏移量
    """
    x1, y1, x2, y2 = roi
    return img[y1:y2, x1:x2], x1, y1


def detect_edges(roi_img: np.ndarray, params=DEFAULT_DETECTION_PARAMETERS) -> np.ndarray:
    """
    边缘检测流水线

    1. 高斯模糊去噪
    2. Canny 边缘检测
    """
    params = params or DEFAULT_DETECTION_PARAMETERS
    # 高斯模糊
    blurred = cv2.GaussianBlur(roi_img, (params.gaussian_blur_ksize, params.gaussian_blur_ksize), 0)

    # 转灰度
    gray = cv2.cvtColor(blurred, cv2.COLOR_BGR2GRAY)

    # Canny 边缘检测
    edges = cv2.Canny(gray, params.canny_low, params.canny_high)

    return edges


def detect_lines(edges: np.ndarray, params=DEFAULT_DETECTION_PARAMETERS) -> List[np.ndarray]:
    """
    使用概率霍夫变换检测直线

    返回:
        lines: [[x1, y1, x2, y2], ...]
    """
    params = params or DEFAULT_DETECTION_PARAMETERS
    lines = cv2.HoughLinesP(
        edges,
        rho=HOUGH_RHO_RESOLUTION,
        theta=HOUGH_ANGLE_RESOLUTION,
        threshold=params.hough_threshold,
        minLineLength=params.hough_min_line_length,
        maxLineGap=params.hough_max_line_gap
    )

    if lines is None:
        return []

    return [line[0] for line in lines]


def classify_line(line: np.ndarray, roi_width: int, roi_height: int, params=DEFAULT_DETECTION_PARAMETERS) -> Tuple[str, float, float]:
    """
    分类直线为水平或垂直，并计算评分
    """
    params = params or DEFAULT_DETECTION_PARAMETERS
    x1, y1, x2, y2 = line

    # 计算角度
    if x2 == x1:
        angle_deg = 90.0
    else:
        angle_rad = np.arctan2(y2 - y1, x2 - x1)
        angle_deg = np.degrees(angle_rad) % 180

    # 分类
    is_horizontal = abs(angle_deg) < params.horizontal_angle_tolerance or \
                    abs(angle_deg - 180) < params.horizontal_angle_tolerance or \
                    abs(angle_deg - 360) < params.horizontal_angle_tolerance

    is_vertical = abs(angle_deg - 90) < params.vertical_angle_tolerance

    if is_horizontal:
        line_type = 'horizontal'
    elif is_vertical:
        line_type = 'vertical'
    else:
        return 'unknown', 0.0, angle_deg

    # 计算评分
    if line_type == 'horizontal':
        avg_y = (y1 + y2) / 2
        dist_to_top = avg_y
        dist_to_bottom = roi_height - avg_y
        proximity_score = 1.0 - min(dist_to_top, dist_to_bottom) / roi_height
    else:
        avg_x = (x1 + x2) / 2
        dist_to_left = avg_x
        dist_to_right = roi_width - avg_x
        proximity_score = 1.0 - min(dist_to_left, dist_to_right) / roi_width

    # 线长度
    length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
    max_possible_length = np.sqrt(roi_width**2 + roi_height**2)
    length_score = length / max_possible_length

    # 角度质量
    if line_type == 'horizontal':
        angle_deviation = min(abs(angle_deg), abs(angle_deg - 180))
    else:
        angle_deviation = abs(angle_deg - 90)

    angle_score = 1.0 - (angle_deviation / 90.0)

    # 综合评分
    score = params.weight_proximity * proximity_score + \
            params.weight_length * length_score + \
            params.weight_angle * angle_score

    return line_type, score, angle_deg


def find_best_lines(lines: List[np.ndarray], roi_width: int, roi_height: int, params=DEFAULT_DETECTION_PARAMETERS) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float, float]:
    """
    筛选最优的水平线和垂直线
    """
    horizontal_lines = []
    vertical_lines = []

    for line in lines:
        line_type, score, angle = classify_line(line, roi_width, roi_height, params)
        if line_type == 'horizontal':
            horizontal_lines.append((line, score))
        elif line_type == 'vertical':
            vertical_lines.append((line, score))

    # 选择最优线
    best_h = max(horizontal_lines, key=lambda x: x[1]) if horizontal_lines else None
    best_v = max(vertical_lines, key=lambda x: x[1]) if vertical_lines else None

    h_line, h_score = best_h if best_h else (None, 0.0)
    v_line, v_score = best_v if best_v else (None, 0.0)

    return h_line, v_line, h_score, v_score


def line_intersection(line1: np.ndarray, line2: np.ndarray) -> Tuple[float, float]:
    """
    计算两条线的交点
    """
    x1, y1, x2, y2 = line1
    x3, y3, x4, y4 = line2

    denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denom) < 1e-10:
        return ((x1 + x3) / 2, (y1 + y3) / 2)

    px = ((x1*y2 - y1*x2) * (x3 - x4) - (x1 - x2) * (x3*y4 - y3*x4)) / denom
    py = ((x1*y2 - y1*x2) * (y3 - y4) - (y1 - y2) * (x3*y4 - y3*x4)) / denom

    return (px, py)


def refine_corner_subpix(roi_gray: np.ndarray, corner: Tuple[float, float]) -> Tuple[float, float]:
    """
    使用 cornerSubPix 进行亚像素细化
    """
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
                SUBPIX_CRITERIA_MAX_ITER, SUBPIX_CRITERIA_EPSILON)

    corner_array = np.array([[corner]], dtype=np.float32)

    cv2.cornerSubPix(roi_gray, corner_array,
                     (SUBPIX_WINDOW_SIZE, SUBPIX_WINDOW_SIZE),
                     (-1, -1), criteria)

    return (corner_array[0][0][0], corner_array[0][0][1])


def detect_corner_in_roi(roi_img: np.ndarray, corner_idx: int, params=DEFAULT_DETECTION_PARAMETERS) -> Tuple[Tuple[int, int], float, Dict[str, Any]]:
    """
    在单个 ROI 中检测角点
    """
    params = params or DEFAULT_DETECTION_PARAMETERS
    h, w = roi_img.shape[:2]

    debug_info = {
        'roi_size': (w, h),
        'corner_idx': corner_idx,
        'lines_detected': 0,
        'horizontal_lines': 0,
        'vertical_lines': 0,
    }

    # 边缘检测
    edges = detect_edges(roi_img, params=params)

    # 计算边缘像素数
    edge_pixels = cv2.countNonZero(edges)
    debug_info['edge_pixels'] = edge_pixels

    # 直线检测
    lines = detect_lines(edges, params)
    debug_info['lines_detected'] = len(lines)

    if len(lines) == 0:
        fallback_positions = [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]
        return fallback_positions[corner_idx], 0.0, debug_info

    # 筛选最优线
    h_line, v_line, h_score, v_score = find_best_lines(lines, w, h, params)

    debug_info['horizontal_lines'] = sum(1 for l in lines if classify_line(l, w, h, params)[0] == 'horizontal')
    debug_info['vertical_lines'] = sum(1 for l in lines if classify_line(l, w, h, params)[0] == 'vertical')
    debug_info['h_score'] = h_score
    debug_info['v_score'] = v_score

    # 计算角点
    if h_line is not None and v_line is not None:
        corner = line_intersection(h_line, v_line)
        confidence = (h_score + v_score) / 2
    elif h_line is not None:
        x1, y1, x2, y2 = h_line
        avg_y = (y1 + y2) / 2
        if corner_idx in [0, 3]:
            corner = (0, avg_y)
        else:
            corner = (w-1, avg_y)
        confidence = h_score * 0.5
    elif v_line is not None:
        x1, y1, x2, y2 = v_line
        avg_x = (x1 + x2) / 2
        if corner_idx in [0, 1]:
            corner = (avg_x, 0)
        else:
            corner = (avg_x, h-1)
        confidence = v_score * 0.5
    else:
        fallback_positions = [(0, 0), (w-1, 0), (w-1, h-1), (0, h-1)]
        corner = fallback_positions[corner_idx]
        confidence = 0.0

    debug_info['raw_corner'] = corner
    debug_info['raw_confidence'] = confidence

    # 亚像素细化
    if confidence >= params.corner_confidence_threshold:
        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
        try:
            corner = refine_corner_subpix(gray, corner)
            debug_info['subpix_refined'] = True
        except Exception as e:
            debug_info['subpix_error'] = str(e)

    return (int(corner[0]), int(corner[1])), confidence, debug_info


def detect_corners_detailed(img: np.ndarray, shrink_min: int = None,
                            shrink_max: int = None,
                            highlight: int = None,
                            params: DetectionParameters = DEFAULT_DETECTION_PARAMETERS) -> Tuple[
                                List[List[int]], List[float], List[Dict[str, Any]]
                            ]:
    """Run the primary detector once and return corners with diagnostics."""
    params = params or DEFAULT_DETECTION_PARAMETERS
    if highlight is not None:
        params = params.replace(ps_highlight=highlight)
    return detect_corners_v410(
        img, shrink_min, shrink_max, return_details=True, params=params
    )


def detect_corners(img: np.ndarray, shrink_min: int = None, shrink_max: int = None,
                   highlight: int = None,
                   params: DetectionParameters = DEFAULT_DETECTION_PARAMETERS) -> List[List[int]]:
    """
    v4.10 角点检测主函数（当前主算法）- 置信度驱动几何推理

    重要说明：
    1. PS 色阶仅用于辅助角点检测，不会修改输入的原图
    2. 返回的角点坐标对应原图尺寸，可直接用于原图裁剪
    3. 调用方应使用原图进行最终的透视裁剪
    4. 使用v4.10置信度驱动几何推理算法

    参数:
        img: 输入图像
        shrink_min: 兼容旧接口
        shrink_max: 兼容旧接口
        highlight: PS 色阶白点阈值（默认 200）

    返回:
        corners: 四角坐标列表 [[x,y], [x,y], [x,y], [x,y]]
    """
    corners, _, _ = detect_corners_detailed(
        img, shrink_min, shrink_max, highlight, params
    )
    return corners


def detect_corners_v48(img: np.ndarray, shrink_min: int = None, shrink_max: int = None,
                        highlight: int = None,
                        params: DetectionParameters = DEFAULT_DETECTION_PARAMETERS) -> List[List[int]]:
    """
    v4.8 角点检测 - 检测失败时扩展搜索

    改进:
    当ROI内检测不到足够直线时，扩大ROI范围重新搜索

    参数:
        img: 输入图像
        shrink_min: 兼容旧接口
        shrink_max: 兼容旧接口
        highlight: PS 色阶白点阈值

    返回:
        corners: 四角坐标列表 [[x,y], [x,y], [x,y], [x,y]]
    """
    params = params or DEFAULT_DETECTION_PARAMETERS
    if highlight is not None:
        params = params.replace(ps_highlight=highlight)
    # PS 色阶预处理
    img_processed = ps_levels(img, highlight=params.ps_highlight)
    h, w = img_processed.shape[:2]

    # 扩展搜索的ROI比例（从默认10%开始，逐步扩大）
    roi_scales = list(params.roi_scales)

    corners = []

    for corner_idx in range(4):
        corner = None

        for scale in roi_scales:
            roi_w = int(w * scale)
            roi_h = int(h * scale)

            # 计算该角点的ROI区域
            if corner_idx == 0:  # 左上
                roi = (0, 0, roi_w, roi_h)
            elif corner_idx == 1:  # 右上
                roi = (w - roi_w, 0, w, roi_h)
            elif corner_idx == 2:  # 右下
                roi = (w - roi_w, h - roi_h, w, h)
            else:  # 左下
                roi = (0, h - roi_h, roi_w, h)

            # 提取ROI并检测
            roi_img, x_offset, y_offset = extract_roi(img_processed, roi)
            corner_roi, confidence, debug_info = detect_corner_in_roi(roi_img, corner_idx, params)

            # 检查是否检测成功（有足够的水平和垂直线）
            h_lines = debug_info.get('horizontal_lines', 0)
            v_lines = debug_info.get('vertical_lines', 0)

            if h_lines > 0 and v_lines > 0 and confidence > 0.3:
                # 检测成功，使用当前结果
                corner = (corner_roi[0] + x_offset, corner_roi[1] + y_offset)
                break

        if corner is None:
            # 所有比例都失败，使用最后一个比例的结果
            fallback_scale = roi_scales[-1]
            roi_w = int(w * fallback_scale)
            roi_h = int(h * fallback_scale)
            if corner_idx == 0:
                roi = (0, 0, roi_w, roi_h)
            elif corner_idx == 1:
                roi = (w - roi_w, 0, w, roi_h)
            elif corner_idx == 2:
                roi = (w - roi_w, h - roi_h, w, h)
            else:
                roi = (0, h - roi_h, roi_w, h)

            roi_img, x_offset, y_offset = extract_roi(img_processed, roi)
            corner_roi, _, _ = detect_corner_in_roi(roi_img, corner_idx, params)
            corner = (corner_roi[0] + x_offset, corner_roi[1] + y_offset)

        corners.append([int(corner[0]), int(corner[1])])

    return corners


def calculate_detection_confidence(debug_info: Dict[str, Any], expansion_level: int = 0) -> float:
    """
    基于检测过程信息计算角点置信度

    参数:
        debug_info: detect_corner_in_roi返回的调试信息
        expansion_level: 扩展搜索层级（0=10%, 1=15%, 2=20%, 3=25%）

    返回:
        0-1之间的置信度值
    """
    # fallback → 0
    if debug_info.get('lines_detected', 0) == 0:
        return 0.0

    confidence = 0.0

    # 1. 边缘丰富度（0-0.25）
    edge_pixels = debug_info.get('edge_pixels', 0)
    confidence += min(edge_pixels / 3000, 0.25)

    # 2. 直线数量（0-0.25）
    lines = debug_info.get('lines_detected', 0)
    confidence += min(lines / 30, 0.25)

    # 3. 最佳线质量（0-0.3）
    h_score = debug_info.get('h_score', 0)
    v_score = debug_info.get('v_score', 0)
    if h_score > 0 and v_score > 0:
        confidence += (h_score + v_score) / 2 * 0.3
    elif h_score > 0 or v_score > 0:
        confidence += max(h_score, v_score) * 0.15

    # 4. 扩展搜索惩罚（每级-0.05）
    confidence -= expansion_level * 0.05

    return max(0.0, min(1.0, confidence))


def detect_corners_v48_with_confidence(img: np.ndarray, shrink_min: int = None,
                                        shrink_max: int = None,
                                        highlight: int = None,
                                        params: DetectionParameters = DEFAULT_DETECTION_PARAMETERS) -> Tuple[List[List[int]], List[float], List[Dict]]:
    """
    v4.8扩展搜索，同时返回每个角点的置信度

    返回:
        (corners, confidences, debug_infos)
    """
    params = params or DEFAULT_DETECTION_PARAMETERS
    if highlight is not None:
        params = params.replace(ps_highlight=highlight)
    # PS 色阶预处理
    img_processed = ps_levels(img, highlight=params.ps_highlight)
    h, w = img_processed.shape[:2]

    # 扩展搜索的ROI比例（从默认10%开始，逐步扩大）
    roi_scales = list(params.roi_scales)

    corners = []
    confidences = []
    debug_infos = []

    for corner_idx in range(4):
        corner = None
        corner_confidence = 0.0
        corner_debug_info = {}
        expansion_level_used = 0

        for level, scale in enumerate(roi_scales):
            roi_w = int(w * scale)
            roi_h = int(h * scale)

            # 计算该角点的ROI区域
            if corner_idx == 0:  # 左上
                roi = (0, 0, roi_w, roi_h)
            elif corner_idx == 1:  # 右上
                roi = (w - roi_w, 0, w, roi_h)
            elif corner_idx == 2:  # 右下
                roi = (w - roi_w, h - roi_h, w, h)
            else:  # 左下
                roi = (0, h - roi_h, roi_w, h)

            # 提取ROI并检测
            roi_img, x_offset, y_offset = extract_roi(img_processed, roi)
            corner_roi, _, debug_info = detect_corner_in_roi(roi_img, corner_idx, params)

            # 检查是否检测成功（有足够的水平和垂直线）
            h_lines = debug_info.get('horizontal_lines', 0)
            v_lines = debug_info.get('vertical_lines', 0)
            raw_confidence = debug_info.get('raw_confidence', 0)

            if h_lines > 0 and v_lines > 0 and raw_confidence > 0.3:
                # 检测成功，使用当前结果
                corner = (corner_roi[0] + x_offset, corner_roi[1] + y_offset)
                corner_debug_info = debug_info
                expansion_level_used = level
                break

        if corner is None:
            # 所有比例都失败，使用最后一个比例的结果
            fallback_scale = roi_scales[-1]
            roi_w = int(w * fallback_scale)
            roi_h = int(h * fallback_scale)
            if corner_idx == 0:
                roi = (0, 0, roi_w, roi_h)
            elif corner_idx == 1:
                roi = (w - roi_w, 0, w, roi_h)
            elif corner_idx == 2:
                roi = (w - roi_w, h - roi_h, w, h)
            else:
                roi = (0, h - roi_h, roi_w, h)

            roi_img, x_offset, y_offset = extract_roi(img_processed, roi)
            corner_roi, _, debug_info = detect_corner_in_roi(roi_img, corner_idx, params)
            corner = (corner_roi[0] + x_offset, corner_roi[1] + y_offset)
            corner_debug_info = debug_info
            expansion_level_used = len(roi_scales) - 1

        # 计算最终置信度
        corner_confidence = calculate_detection_confidence(corner_debug_info, expansion_level_used)

        corners.append([int(corner[0]), int(corner[1])])
        confidences.append(corner_confidence)
        corner_debug_info['expansion_level'] = expansion_level_used
        debug_infos.append(corner_debug_info)

    return corners, confidences, debug_infos


def calculate_interior_angles(corners):
    """
    计算四边形四角的内角（度）

    参数:
        corners: [[x,y], [x,y], [x,y], [x,y]] 按左上、右上、右下、左下排序

    返回:
        angles: [angle1, angle2, angle3, angle4] 对应四个角的内角
    """
    import math

    def vector_angle(v1, v2):
        """计算两个向量的夹角（度）"""
        dot = v1[0]*v2[0] + v1[1]*v2[1]
        norm1 = math.sqrt(v1[0]**2 + v1[1]**2)
        norm2 = math.sqrt(v2[0]**2 + v2[1]**2)

        if norm1 == 0 or norm2 == 0:
            return 0

        cos_angle = dot / (norm1 * norm2)
        # 限制在[-1, 1]范围内避免数值误差
        cos_angle = max(-1, min(1, cos_angle))
        return math.degrees(math.acos(cos_angle))

    angles = []
    n = len(corners)

    for i in range(n):
        # 当前角点
        curr = corners[i]
        # 前一个角点
        prev = corners[(i-1) % n]
        # 后一个角点
        next_pt = corners[(i+1) % n]

        # 向量：从前一个角点指向当前角点
        v1 = (curr[0] - prev[0], curr[1] - prev[1])
        # 向量：从当前角点指向后一个角点
        v2 = (next_pt[0] - curr[0], next_pt[1] - curr[1])

        angle = vector_angle(v1, v2)
        angles.append(angle)

    return angles


def validate_corners_by_angle(corners, tolerance=20):
    """
    通过内角验证角点质量，识别异常角点

    参数:
        corners: 四角坐标 [[x,y], ...] 按左上、右上、右下、左下排序
        tolerance: 角度容差，90±tolerance视为正常（默认20°）

    返回:
        (status, anomaly_indices)
        status: 'all_good' | 'one_anomaly' | 'opposite_anomaly' | 'uncertain'
        anomaly_indices: 异常角点的索引列表
    """
    angles = calculate_interior_angles(corners)

    # 判断每个角是否正常
    normal_mask = [abs(a - 90) <= tolerance for a in angles]
    normal_count = sum(normal_mask)

    # 找出异常角点
    anomaly_indices = [i for i, is_normal in enumerate(normal_mask) if not is_normal]

    # 根据正常角数量判断状态
    if normal_count == 4:
        return 'all_good', []
    elif normal_count == 3:
        return 'one_anomaly', anomaly_indices
    elif normal_count == 2:
        # 检查是否是对角异常
        normal_indices = [i for i, is_normal in enumerate(normal_mask) if is_normal]
        if len(normal_indices) == 2:
            diff = abs(normal_indices[0] - normal_indices[1])
            if diff == 1 or diff == 3:  # 相邻（索引差1或3表示0和3相邻）
                return 'opposite_anomaly', anomaly_indices
        return 'multiple_anomalies', anomaly_indices
    else:
        return 'uncertain', anomaly_indices


def infer_missing_corner(corners, known_indices):
    """
    根据3个已知角点，利用矩形对角线互相平分特性推理第4个角点

    原理:
    - 矩形对角线AC的中点 = 对角线BD的中点
    - 若已知A, B, C，则 D = B + C - A （向量运算）

    参数:
        corners: 四角坐标列表，其中3个是准确的，1个可以任意值
        known_indices: 已知准确角点的索引列表 [i, j, k]，长度为3

    返回:
        inferred_corner: [x, y] 推理得到的第4个角点坐标

    注意:
        - 假设3个已知角点构成直角（照片被压平）
    """
    assert len(known_indices) == 3, "需要恰好3个已知角点"

    # 找出缺失的角点索引
    all_indices = {0, 1, 2, 3}
    missing_index = list(all_indices - set(known_indices))[0]

    # 根据缺失角的索引，选择推理方法
    corners_dict = {i: corners[i] for i in known_indices}

    if missing_index == 0:  # 缺左上，已知右上、右下、左下
        # A = B + D - C
        inferred = [
            corners_dict[1][0] + corners_dict[3][0] - corners_dict[2][0],
            corners_dict[1][1] + corners_dict[3][1] - corners_dict[2][1]
        ]
    elif missing_index == 1:  # 缺右上，已知左上、右下、左下
        # B = A + C - D
        inferred = [
            corners_dict[0][0] + corners_dict[2][0] - corners_dict[3][0],
            corners_dict[0][1] + corners_dict[2][1] - corners_dict[3][1]
        ]
    elif missing_index == 2:  # 缺右下，已知左上、右上、左下
        # C = B + D - A
        inferred = [
            corners_dict[1][0] + corners_dict[3][0] - corners_dict[0][0],
            corners_dict[1][1] + corners_dict[3][1] - corners_dict[0][1]
        ]
    else:  # missing_index == 3, 缺左下，已知左上、右上、右下
        # D = A + C - B
        inferred = [
            corners_dict[0][0] + corners_dict[2][0] - corners_dict[1][0],
            corners_dict[0][1] + corners_dict[2][1] - corners_dict[1][1]
        ]

    return inferred


def infer_by_edge_and_diagonal(corners, confidences, img_width, img_height, confidence_threshold=0.6):
    """
    已知一条高置信度边 + 对角高置信度点，推理第4个点（带边界约束）

    参数:
        corners: 四角坐标 [[x,y], [x,y], [x,y], [x,y]]
        confidences: 四个角点的置信度列表
        img_width: 图片宽度
        img_height: 图片高度
        confidence_threshold: 置信度阈值，默认0.6

    返回:
        (inferred_corner, confidence, target_idx)
        inferred_corner: 推理得到的角点坐标 [x, y] 或 None
        confidence: 推理置信度
        target_idx: 待推理角点的索引或 None
    """
    best_inference = None
    best_conf = 0.0
    best_idx = None

    # 最大允许超出比例
    MAX_VIOLATION_RATIO = 0.1

    # 尝试4条边
    for edge_idx in range(4):
        i1 = edge_idx
        i2 = (edge_idx + 1) % 4
        i3 = (edge_idx + 2) % 4  # 对角
        i4 = (edge_idx + 3) % 4  # 待推理

        # 检查条件：边两端 + 对角都高置信度，待推理点低置信度
        if (confidences[i1] > confidence_threshold and confidences[i2] > confidence_threshold and
            confidences[i3] > confidence_threshold and confidences[i4] <= confidence_threshold):

            # 平行四边形法则：i4 = i1 + (i3 - i2)
            inferred = [
                int(corners[i1][0] + corners[i3][0] - corners[i2][0]),
                int(corners[i1][1] + corners[i3][1] - corners[i2][1])
            ]

            # 检查边界违规程度
            violation = calculate_boundary_violation(inferred, img_width, img_height)

            if violation > MAX_VIOLATION_RATIO:
                # 严重超出，放弃此推理
                continue

            # 轻微超出则截断到边界
            clamped = clamp_corner_to_boundary(inferred, img_width, img_height, margin=0)

            # 计算推理置信度（边界截断降低置信度）
            inferred_conf = (confidences[i1] + confidences[i2] + confidences[i3]) / 3 * 0.9
            if violation > 0:
                boundary_penalty = min(violation * 2, 0.3)
                inferred_conf *= (1 - boundary_penalty)

            if inferred_conf > best_conf:
                best_conf = inferred_conf
                best_inference = clamped
                best_idx = i4

    return best_inference, best_conf, best_idx


def clamp_corner_to_boundary(corner, img_width, img_height, margin=0):
    """
    将角点坐标限制在图片边界内

    参数:
        corner: [x, y] 角点坐标
        img_width: 图片宽度
        img_height: 图片高度
        margin: 边距（默认0，可留出血边）

    返回:
        clamped_corner: 限制后的坐标 [x, y]
    """
    x = max(margin, min(corner[0], img_width - 1 - margin))
    y = max(margin, min(corner[1], img_height - 1 - margin))
    return [int(x), int(y)]


def is_corner_in_boundary(corner, img_width, img_height, margin=0):
    """
    检查角点是否在图片边界内

    参数:
        corner: [x, y] 角点坐标
        img_width: 图片宽度
        img_height: 图片高度
        margin: 边距

    返回:
        True if 角点在边界内
    """
    return (margin <= corner[0] <= img_width - 1 - margin and
            margin <= corner[1] <= img_height - 1 - margin)


def calculate_boundary_violation(corner, img_width, img_height, margin=0):
    """
    计算角点超出边界的程度

    参数:
        margin: 边距

    返回:
        violation_ratio: 超出比例（0表示在边界内，>0表示超出）
    """
    dx = 0
    dy = 0

    if corner[0] < margin:
        dx = margin - corner[0]
    elif corner[0] > img_width - 1 - margin:
        dx = corner[0] - (img_width - 1 - margin)

    if corner[1] < margin:
        dy = margin - corner[1]
    elif corner[1] > img_height - 1 - margin:
        dy = corner[1] - (img_height - 1 - margin)

    # 计算超出比例（相对于图片尺寸）
    if img_width > 0 and img_height > 0:
        return (dx / img_width + dy / img_height)
    return float('inf')


def infer_by_two_edges(corners, confidences, img_width, img_height, confidence_threshold=0.6):
    """
    已知两条相邻高置信度边（3个角点），推理第4个角点（带边界约束）

    参数:
        corners: 四角坐标 [[x,y], [x,y], [x,y], [x,y]]
        confidences: 四个角点的置信度列表
        img_width: 图片宽度
        img_height: 图片高度
        confidence_threshold: 置信度阈值，默认0.6

    返回:
        (inferred_corner, confidence, target_idx)
        inferred_corner: 推理得到的角点坐标 [x, y] 或 None
        confidence: 推理置信度
        target_idx: 缺失角点的索引或 None
    """
    # 找所有高置信度边
    high_conf_edges = []
    for i in range(4):
        j = (i + 1) % 4
        if confidences[i] > confidence_threshold and confidences[j] > confidence_threshold:
            high_conf_edges.append((i, j))

    if len(high_conf_edges) < 2:
        return None, 0.0, None

    # 最大允许超出比例
    MAX_VIOLATION_RATIO = 0.1

    # 检查每对边是否相邻
    for i, edge1 in enumerate(high_conf_edges):
        for edge2 in high_conf_edges[i+1:]:
            all_corners = set([edge1[0], edge1[1], edge2[0], edge2[1]])

            # 相邻边包含3个不同角点
            if len(all_corners) == 3:
                missing_idx = ({0, 1, 2, 3} - all_corners).pop()

                # 使用infer_missing_corner函数
                known_indices = list(all_corners)
                inferred = infer_missing_corner(corners, known_indices)

                # 边界检查
                violation = calculate_boundary_violation(inferred, img_width, img_height)

                if violation > MAX_VIOLATION_RATIO:
                    # 严重超出，放弃此推理
                    continue

                # 轻微超出则截断到边界
                clamped = clamp_corner_to_boundary(inferred, img_width, img_height, margin=0)

                # 置信度（边界截断降低置信度）
                conf = sum(confidences[i] for i in all_corners) / 3 * 0.95
                if violation > 0:
                    boundary_penalty = min(violation * 2, 0.3)
                    conf *= (1 - boundary_penalty)

                return clamped, conf, missing_idx

    return None, 0.0, None


def detect_corners_v410(img: np.ndarray, shrink_min: int = None, shrink_max: int = None,
                        highlight: int = None,
                        confidence_threshold: float = None,
                        return_details: bool = False,
                        params: DetectionParameters = DEFAULT_DETECTION_PARAMETERS):
    """
    v4.10/v5.0 角点检测 - 置信度驱动几何推理（带边界约束）

    改进:
    1. 先用v4.8扩展搜索检测四角，同时获取置信度
    2. 如果存在高置信度边，尝试几何推理修正低置信度角点
    3. 推理结果通过边界约束确保在图片范围内
    4. 验证推理后的几何一致性

    参数:
        img: 输入图像
        confidence_threshold: 高置信度阈值（默认0.6）

    返回:
        corners: 四角坐标列表 [[x,y], [x,y], [x,y], [x,y]]
    """
    import numpy as np

    params = params or DEFAULT_DETECTION_PARAMETERS
    if highlight is not None:
        params = params.replace(ps_highlight=highlight)
    if confidence_threshold is None:
        confidence_threshold = params.confidence_threshold

    h, w = img.shape[:2]  # 获取图片尺寸

    # Step 1: v4.8检测 + 置信度
    corners, confidences, debug_infos = detect_corners_v48_with_confidence(
        img, shrink_min, shrink_max, params=params
    )

    def finish():
        for index in range(4):
            corners[index] = clamp_corner_to_boundary(corners[index], w, h, margin=0)
        if return_details:
            return corners, confidences, debug_infos
        return corners

    # Step 2: 检查是否全部高置信度
    if all(c > confidence_threshold for c in confidences):
        return finish()

    # Step 3: 尝试双边推理（传入图片尺寸）
    inferred, inf_conf, target_idx = infer_by_two_edges(
        corners, confidences, w, h, confidence_threshold
    )
    if inferred and inf_conf > confidences[target_idx]:
        corners[target_idx] = inferred
        confidences[target_idx] = inf_conf

        # 验证更新后是否满足几何约束
        angles = calculate_interior_angles(corners)
        if all(60 < a < 120 for a in angles):  # 宽松角度检查
            return finish()

    # Step 4: 尝试单边+对角推理（传入图片尺寸）
    inferred, inf_conf, target_idx = infer_by_edge_and_diagonal(
        corners, confidences, w, h, confidence_threshold
    )
    if inferred and inf_conf > confidences[target_idx]:
        corners[target_idx] = inferred
        confidences[target_idx] = inf_conf

        # 验证角度
        angles = calculate_interior_angles(corners)
        if all(60 < a < 120 for a in angles):
            return finish()

    # Step 5: 如果推理都失败或验证不通过，返回v4.8结果

    return finish()


def detect_corners_v49(img: np.ndarray, shrink_min: int = None, shrink_max: int = None,
                        highlight: int = PS_LEVELS_HIGHLIGHT, params=None,
                        angle_tolerance: int = 20) -> List[List[int]]:
    """
    v4.9 角点检测 - 几何约束验证与推理

    改进:
    1. 先用v4.8扩展搜索检测四角
    2. 通过角度验证识别异常角点
    3. 如果3个角点正常，用几何推理修正第4个异常角点
    4. 返回修正后的结果

    参数:
        img: 输入图像
        shrink_min: 兼容旧接口
        shrink_max: 兼容旧接口
        highlight: PS 色阶白点阈值
        angle_tolerance: 角度容差（度），默认20°

    返回:
        corners: 四角坐标列表 [[x,y], [x,y], [x,y], [x,y]]
    """
    # Step 1: 先用v4.8检测（带扩展搜索）
    corners = detect_corners_v48(img, shrink_min, shrink_max, highlight)

    # Step 2: 角度验证
    status, anomaly_indices = validate_corners_by_angle(corners, tolerance=angle_tolerance)

    # Step 3: 根据验证结果处理
    if status == 'all_good':
        # 四角都正常，直接返回
        return corners

    elif status == 'one_anomaly' and len(anomaly_indices) == 1:
        # 只有1个异常角，可以用几何推理修正
        missing_idx = anomaly_indices[0]
        known_indices = [i for i in range(4) if i != missing_idx]

        # 推理缺失的角点
        inferred = infer_missing_corner(corners, known_indices)

        # 替换异常角点
        corners[missing_idx] = inferred

        return corners

    elif status == 'opposite_anomaly' and len(anomaly_indices) == 2:
        # 对角异常，尝试两种推理方案，选择更优的
        # 方案1: 假设第一个异常角是错的，用其他3个推理
        # 方案2: 假设第二个异常角是错的，用其他3个推理

        idx1, idx2 = anomaly_indices

        # 方案1
        known1 = [i for i in range(4) if i != idx1]
        inferred1 = infer_missing_corner(corners, known1)
        corners1 = corners.copy()
        corners1[idx1] = inferred1
        angles1 = calculate_interior_angles(corners1)
        score1 = sum(abs(a - 90) for a in angles1)  # 越接近90°分数越低越好

        # 方案2
        known2 = [i for i in range(4) if i != idx2]
        inferred2 = infer_missing_corner(corners, known2)
        corners2 = corners.copy()
        corners2[idx2] = inferred2
        angles2 = calculate_interior_angles(corners2)
        score2 = sum(abs(a - 90) for a in angles2)

        # 选择使四角更接近矩形的方案
        if score1 < score2:
            return corners1
        else:
            return corners2

    else:
        # 其他情况（多个异常/不确定），返回原结果
        return corners


def test_confidence_calculation():
    """测试置信度计算函数"""
    print("\n=== 置信度计算测试 ===\n")

    # 测试用例1: 高置信度（边缘丰富、线多、质量好）
    high_confidence_debug = {
        'lines_detected': 35,
        'edge_pixels': 4000,
        'h_score': 0.9,
        'v_score': 0.85
    }
    conf_high = calculate_detection_confidence(high_confidence_debug, expansion_level=0)
    print(f"高置信度测试:")
    print(f"  输入: lines=35, edge_pixels=4000, h_score=0.9, v_score=0.85, level=0")
    print(f"  输出: {conf_high:.4f} (期望 > 0.7)")
    assert conf_high > 0.7, f"高置信度应 > 0.7, 实际 {conf_high}"

    # 测试用例2: 中等置信度
    mid_confidence_debug = {
        'lines_detected': 20,
        'edge_pixels': 2000,
        'h_score': 0.6,
        'v_score': 0.5
    }
    conf_mid = calculate_detection_confidence(mid_confidence_debug, expansion_level=1)
    print(f"\n中等置信度测试:")
    print(f"  输入: lines=20, edge_pixels=2000, h_score=0.6, v_score=0.5, level=1")
    print(f"  输出: {conf_mid:.4f} (期望 0.5-0.7)")
    assert 0.5 < conf_mid < 0.7, f"中等置信度应在 0.5-0.7 之间, 实际 {conf_mid}"

    # 测试用例3: 低置信度（边缘少、线少）
    low_confidence_debug = {
        'lines_detected': 5,
        'edge_pixels': 500,
        'h_score': 0.3,
        'v_score': 0.2
    }
    conf_low = calculate_detection_confidence(low_confidence_debug, expansion_level=2)
    print(f"\n低置信度测试:")
    print(f"  输入: lines=5, edge_pixels=500, h_score=0.3, v_score=0.2, level=2")
    print(f"  输出: {conf_low:.4f} (期望 < 0.35)")
    assert conf_low < 0.35, f"低置信度应 < 0.35, 实际 {conf_low}"

    # 测试用例4: fallback情况（无线）
    fallback_debug = {
        'lines_detected': 0,
        'edge_pixels': 100,
        'h_score': 0,
        'v_score': 0
    }
    conf_fallback = calculate_detection_confidence(fallback_debug, expansion_level=0)
    print(f"\nFallback测试:")
    print(f"  输入: lines=0")
    print(f"  输出: {conf_fallback:.4f} (期望 = 0)")
    assert conf_fallback == 0.0, f"fallback 置信度应为 0, 实际 {conf_fallback}"

    # 测试用例5: 只有水平线
    h_only_debug = {
        'lines_detected': 15,
        'edge_pixels': 1500,
        'h_score': 0.8,
        'v_score': 0
    }
    conf_h_only = calculate_detection_confidence(h_only_debug, expansion_level=0)
    print(f"\n仅水平线测试:")
    print(f"  输入: lines=15, h_score=0.8, v_score=0")
    print(f"  输出: {conf_h_only:.4f}")
    assert conf_h_only > 0, f"应有部分置信度, 实际 {conf_h_only}"

    # 测试用例6: 扩展搜索惩罚
    base_debug = {
        'lines_detected': 25,
        'edge_pixels': 3000,
        'h_score': 0.7,
        'v_score': 0.7
    }
    conf_level0 = calculate_detection_confidence(base_debug, expansion_level=0)
    conf_level1 = calculate_detection_confidence(base_debug, expansion_level=1)
    conf_level2 = calculate_detection_confidence(base_debug, expansion_level=2)
    conf_level3 = calculate_detection_confidence(base_debug, expansion_level=3)
    print(f"\n扩展搜索惩罚测试:")
    print(f"  level 0: {conf_level0:.4f}")
    print(f"  level 1: {conf_level1:.4f} (应比 level 0 低 0.05)")
    print(f"  level 2: {conf_level2:.4f} (应比 level 0 低 0.10)")
    print(f"  level 3: {conf_level3:.4f} (应比 level 0 低 0.15)")
    assert abs((conf_level0 - conf_level1) - 0.05) < 0.001, "level 1 应比 level 0 低 0.05"
    assert abs((conf_level0 - conf_level2) - 0.10) < 0.001, "level 2 应比 level 0 低 0.10"
    assert abs((conf_level0 - conf_level3) - 0.15) < 0.001, "level 3 应比 level 0 低 0.15"

    print("\n✓ 所有置信度计算测试通过!")
    return True


if __name__ == "__main__":
    import sys

    # 运行置信度测试
    test_confidence_calculation()

    # 测试正方形
    square = [[0,0], [100,0], [100,100], [0,100]]
    angles = calculate_interior_angles(square)
    print(f"正方形四角: {angles}")  # 应接近 [90, 90, 90, 90]

    # 测试长方形
    rect = [[0,0], [200,0], [200,100], [0,100]]
    angles = calculate_interior_angles(rect)
    print(f"长方形四角: {angles}")  # 应接近 [90, 90, 90, 90]

    # 测试角度验证 - 正常情况
    print("\n=== 角度验证测试 ===")
    rect = [[0,0], [100,0], [100,100], [0,100]]
    status, anomalies = validate_corners_by_angle(rect)
    print(f"正常矩形: {status}, 异常角: {anomalies}")  # 应输出 all_good, []

    # 测试角度验证 - 一角异常（右上偏移）
    distorted = [[0,0], [120,10], [100,100], [0,100]]  # 右上角向右偏移
    status, anomalies = validate_corners_by_angle(distorted)
    print(f"轻微变形: {status}, 异常角: {anomalies}")

    angles = calculate_interior_angles(distorted)
    print(f"变形四角: {[f'{a:.1f}' for a in angles]}")

    # 测试更极端的变形 - 应该触发异常检测
    extreme_distorted = [[0,0], [150,30], [100,100], [0,100]]  # 右上角更大幅度偏移
    status, anomalies = validate_corners_by_angle(extreme_distorted)
    print(f"极端变形: {status}, 异常角: {anomalies}")

    angles = calculate_interior_angles(extreme_distorted)
    print(f"极端变形四角: {[f'{a:.1f}' for a in angles]}")

    # 测试几何推理
    print("\n=== 几何推理测试 ===")

    # 完美矩形，移除一个角测试推理
    rect = [[0,0], [100,0], [100,100], [0,100]]

    # 测试缺右下(2)，已知0,1,3
    inferred = infer_missing_corner(rect, [0, 1, 3])
    print(f"缺右下，推理结果: {inferred}, 实际: {rect[2]}")  # 应接近 [100,100]

    # 测试缺左上(0)，已知1,2,3
    inferred = infer_missing_corner(rect, [1, 2, 3])
    print(f"缺左上，推理结果: {inferred}, 实际: {rect[0]}")  # 应接近 [0,0]

    # 测试缺右上(1)，已知0,2,3
    inferred = infer_missing_corner(rect, [0, 2, 3])
    print(f"缺右上，推理结果: {inferred}, 实际: {rect[1]}")  # 应接近 [100,0]

    # 测试缺左下(3)，已知0,1,2
    inferred = infer_missing_corner(rect, [0, 1, 2])
    print(f"缺左下，推理结果: {inferred}, 实际: {rect[3]}")  # 应接近 [0,100]

    # 测试含噪声的情况
    noisy_rect = [[5,3], [98,2], [103,97], [2,105]]  # 接近矩形但有偏差
    inferred = infer_missing_corner(noisy_rect, [0, 1, 3])
    print(f"含噪声缺右下，推理: {inferred}, 实际近似: [100,100]")

    # 测试基于置信度的推理函数
    print("\n=== 基于置信度的推理测试 ===")

    # 测试双边推理
    print("\n--- 双边推理测试 ---")
    rect = [[0,0], [100,0], [100,100], [0,100]]

    # 场景1: 边0-1和边1-2高置信度（缺左下，索引3）
    confidences = [0.8, 0.85, 0.75, 0.3]  # 0,1,2高，3低
    inferred, conf, idx = infer_by_two_edges(rect, confidences, 101, 101, confidence_threshold=0.6)
    print(f"双边推理(缺左下): 推理结果={inferred}, 置信度={conf:.4f}, 索引={idx}")
    assert inferred is not None, "双边推理应成功"
    assert idx == 3, f"应推理索引3，实际{idx}"
    assert inferred == [0, 100], f"左下应为[0,100]，实际{inferred}"

    # 场景2: 边2-3和边3-0高置信度（缺右上，索引1）
    confidences = [0.8, 0.3, 0.85, 0.9]  # 0,2,3高，1低
    inferred, conf, idx = infer_by_two_edges(rect, confidences, 101, 101, confidence_threshold=0.6)
    print(f"双边推理(缺右上): 推理结果={inferred}, 置信度={conf:.4f}, 索引={idx}")
    assert inferred is not None, "双边推理应成功"
    assert idx == 1, f"应推理索引1，实际{idx}"
    assert inferred == [100, 0], f"右上应为[100,0]，实际{inferred}"

    # 场景3: 只有一条边高置信度（应失败）
    confidences = [0.8, 0.85, 0.3, 0.3]  # 只有边0-1高
    inferred, conf, idx = infer_by_two_edges(rect, confidences, 101, 101, confidence_threshold=0.6)
    print(f"单边高置信度: 推理结果={inferred}, 置信度={conf:.4f}, 索引={idx}")
    assert inferred is None, "单边高置信度应无法推理"

    # 测试单边+对角推理
    print("\n--- 单边+对角推理测试 ---")

    # 场景1: 边0-1高 + 对角2高，缺3
    confidences = [0.8, 0.85, 0.75, 0.3]  # 0,1,2高，3低
    inferred, conf, idx = infer_by_edge_and_diagonal(rect, confidences, 101, 101, confidence_threshold=0.6)
    print(f"边0-1+对角2: 推理结果={inferred}, 置信度={conf:.4f}, 索引={idx}")
    assert inferred is not None, "单边+对角推理应成功"
    assert idx == 3, f"应推理索引3，实际{idx}"
    assert inferred == [0, 100], f"左下应为[0,100]，实际{inferred}"

    # 场景2: 边1-2高 + 对角3高，缺0
    confidences = [0.3, 0.85, 0.8, 0.75]  # 1,2,3高，0低
    inferred, conf, idx = infer_by_edge_and_diagonal(rect, confidences, 101, 101, confidence_threshold=0.6)
    print(f"边1-2+对角3: 推理结果={inferred}, 置信度={conf:.4f}, 索引={idx}")
    assert inferred is not None, "单边+对角推理应成功"
    assert idx == 0, f"应推理索引0，实际{idx}"
    assert inferred == [0, 0], f"左上应为[0,0]，实际{inferred}"

    # 场景3: 边2-3高 + 对角0高，缺1
    confidences = [0.75, 0.3, 0.85, 0.8]  # 0,2,3高，1低
    inferred, conf, idx = infer_by_edge_and_diagonal(rect, confidences, 101, 101, confidence_threshold=0.6)
    print(f"边2-3+对角0: 推理结果={inferred}, 置信度={conf:.4f}, 索引={idx}")
    assert inferred is not None, "单边+对角推理应成功"
    assert idx == 1, f"应推理索引1，实际{idx}"
    assert inferred == [100, 0], f"右上应为[100,0]，实际{inferred}"

    # 场景4: 边3-0高 + 对角1高，缺2
    confidences = [0.8, 0.75, 0.3, 0.85]  # 0,1,3高，2低
    inferred, conf, idx = infer_by_edge_and_diagonal(rect, confidences, 101, 101, confidence_threshold=0.6)
    print(f"边3-0+对角1: 推理结果={inferred}, 置信度={conf:.4f}, 索引={idx}")
    assert inferred is not None, "单边+对角推理应成功"
    assert idx == 2, f"应推理索引2，实际{idx}"
    assert inferred == [100, 100], f"右下应为[100,100]，实际{inferred}"

    # 场景5: 条件不满足（没有对角高置信度）
    confidences = [0.8, 0.85, 0.3, 0.3]  # 只有边0-1高，对角2低
    inferred, conf, idx = infer_by_edge_and_diagonal(rect, confidences, 101, 101, confidence_threshold=0.6)
    print(f"条件不满足: 推理结果={inferred}, 置信度={conf:.4f}, 索引={idx}")
    assert inferred is None, "条件不满足时应无法推理"

    # 场景6: 阈值测试
    print("\n--- 阈值测试 ---")
    confidences = [0.55, 0.55, 0.55, 0.3]  # 都低于0.6阈值
    inferred, conf, idx = infer_by_two_edges(rect, confidences, 101, 101, confidence_threshold=0.6)
    print(f"阈值0.6，置信度0.55: 推理结果={inferred}")
    assert inferred is None, "低于阈值应无法推理"

    confidences = [0.65, 0.65, 0.65, 0.3]  # 都高于0.6阈值
    inferred, conf, idx = infer_by_two_edges(rect, confidences, 101, 101, confidence_threshold=0.6)
    print(f"阈值0.6，置信度0.65: 推理结果={inferred}, 索引={idx}")
    assert inferred is not None, "高于阈值应能推理"

    print("\n✓ 所有基于置信度的推理测试通过!")

    print()

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)

        if img is not None:
            corners = detect_corners(img)

            print("检测到的角点:")
            corner_names = ["左上", "右上", "右下", "左下"]
            for i, corner in enumerate(corners):
                print(f"  {corner_names[i]}: {corner}")

            # 可视化
            h, w = img.shape[:2]
            scale = 800 / max(h, w)
            display = cv2.resize(img, (int(w*scale), int(h*scale)))

            for corner in corners:
                x, y = int(corner[0] * scale), int(corner[1] * scale)
                cv2.circle(display, (x, y), 5, (0, 0, 255), -1)

            cv2.imshow("Corners", display)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"无法读取图片: {img_path}")
    else:
        print("用法: python corner_detector_v4.py <图片路径>")
