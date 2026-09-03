# -*- coding: utf-8 -*-
"""
透视变换模块 - 四点透视校正
"""

import cv2
import numpy as np


def order_points(pts):
    """
    将四角按顺序排列：左上、右上、右下、左下

    参数:
        pts: 四角坐标 [[x,y], [x,y], [x,y], [x,y]]

    返回:
        ordered_pts: 排序后的坐标
    """
    pts = np.array(pts, dtype=np.float32)

    # 按 x+y 排序，最小为左上，最大为右下
    s = pts.sum(axis=1)
    ordered = np.zeros_like(pts)
    ordered[0] = pts[np.argmin(s)]  # 左上
    ordered[2] = pts[np.argmax(s)]  # 右下

    # 按 x-y 排序
    d = pts[:, 0] - pts[:, 1]
    ordered[1] = pts[np.argmax(d)]  # 右上 (x大, y小 → x-y最大)
    ordered[3] = pts[np.argmin(d)]  # 左下 (x小, y大 → x-y最小)

    return ordered


def compute_output_size(pts):
    """
    计算透视变换后的输出尺寸

    参数:
        pts: 排序后的四角坐标

    返回:
        width: 输出宽度
        height: 输出高度
    """
    (tl, tr, br, bl) = pts

    # 计算宽度（上下边的最大值）
    width_a = np.sqrt((br[0] - bl[0])**2 + (br[1] - bl[1])**2)
    width_b = np.sqrt((tr[0] - tl[0])**2 + (tr[1] - tl[1])**2)
    width = int(max(width_a, width_b))

    # 计算高度（左右边的高度）
    height_a = np.sqrt((tr[0] - br[0])**2 + (tr[1] - br[1])**2)
    height_b = np.sqrt((tl[0] - bl[0])**2 + (tl[1] - bl[1])**2)
    height = int(max(height_a, height_b))

    return width, height


def perspective_transform(img, corners):
    """
    对图像进行透视变换

    参数:
        img: 输入图像
        corners: 四角坐标 [[x,y], [x,y], [x,y], [x,y]]

    返回:
        warped: 透视变换后的图像
    """
    pts = order_points(corners)
    (tl, tr, br, bl) = pts

    width, height = compute_output_size(pts)

    # 目标点
    dst = np.array([
        [0, 0],
        [width - 1, 0],
        [width - 1, height - 1],
        [0, height - 1]
    ], dtype=np.float32)

    # 计算透视变换矩阵
    M = cv2.getPerspectiveTransform(pts, dst)

    # 应用透视变换
    warped = cv2.warpPerspective(img, M, (width, height))

    return warped
