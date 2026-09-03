# -*- coding: utf-8 -*-
"""
photocut 配置参数
集中管理所有可调参数
"""

import numpy as np

# HSV 白色检测阈值
WHITE_SATURATION_MAX = 30  # 饱和度上限（低于此值视为白色候选）
WHITE_BRIGHTNESS_MIN = 200  # 亮度下限（高于此值视为白色候选）

# 角点搜索参数
SOBEL_KSIZE = 3  # Sobel 算子核大小
EDGE_SEARCH_STEP = 5  # 边缘搜索步长（像素）
EDGE_SEARCH_MAX_ITER = 100  # 最大搜索迭代次数

# 角点收缩参数
CORNER_SHRINK_MIN = 25  # 最小收缩量（像素）
CORNER_SHRINK_MAX = 70  # 最大收缩量（像素）
CORNER_SHRINK_DEFAULT = 40  # 默认收缩量（像素）

# 预览图参数
PREVIEW_MAX_SIZE = 1200  # 预览图最大边长

# 内存优化
THUMB_CACHE_SIZE = 20  # 缩略图内存缓存数量
BATCH_SIZE = 30  # 每批处理照片数量

# 文件名
CORNERS_INFO_FILE = "corners_info.json"
THUMBS_DIR = ".thumbs"
OUTPUT_DIR = "裁切成品"

# 本地数据集归档
DATASET_SCHEMA_VERSION = 1
SCHEMA_VERSION = DATASET_SCHEMA_VERSION
BATCH_REFERENCE_FILE = ".photocut_batch.json"
ALGORITHM_VERSION = "5.2"
V7_ALGORITHM_VERSION = "7.1"
# Detector, selector, and GUI versions are intentionally independent.  The
# stable ``auto-v4`` record identity remains compatible with existing batches.
DEFAULT_DETECTOR = "auto"
DEFAULT_SCENE_PROFILE = "scanner_white"

# ========== v4.1 边缘检测参数 ==========
# 重要说明：
# - PS 色阶仅用于角点检测辅助，不会修改原图
# - 检测返回的坐标对应原图尺寸
# - 最终裁剪时必须使用原图，不能用色阶后的图

# ROI 设置
ROI_SCALE = 0.10  # 四角 ROI 比例（10%，v4.1 最佳配置）

# 预处理
GAUSSIAN_BLUR_KSIZE = 5  # 高斯模糊核大小

# PS 色阶参数（经测试优化）
PS_LEVELS_HIGHLIGHT = 230  # 白点阈值（默认 200，优化后 230）

# Canny 边缘检测 (v4.1: 最佳配置)
CANNY_LOW_THRESHOLD = 30
CANNY_HIGH_THRESHOLD = 90

# Hough 直线检测 (v5: 优化后参数)
HOUGH_RHO_RESOLUTION = 1  # 距离分辨率（像素）
HOUGH_ANGLE_RESOLUTION = np.pi / 180  # 角度分辨率（弧度）
HOUGH_THRESHOLD = 35  # 累加器阈值 (v5优化: 35)
HOUGH_MIN_LINE_LENGTH = 40  # 最小线长度 (v5优化: 40)
HOUGH_MAX_LINE_GAP = 25  # 最大线段间隙 (v5优化: 25)

# 直线角度容忍度（度）
HORIZONTAL_ANGLE_TOLERANCE = 10  # 水平线容忍度
VERTICAL_ANGLE_TOLERANCE = 10    # 垂直线容忍度

# 角点置信度阈值
CORNER_CONFIDENCE_THRESHOLD = 0.7  # 触发亚像素细化的阈值
MIN_CORNER_CONFIDENCE = 0.5        # 最低可接受置信度

# 亚像素细化
SUBPIX_WINDOW_SIZE = 11  # 搜索窗口大小
SUBPIX_CRITERIA_MAX_ITER = 30
SUBPIX_CRITERIA_EPSILON = 0.001

# 评分权重 (v5: 优化后权重)
WEIGHT_PROXIMITY = 0.5  # 靠近外侧边缘权重 (v5优化: 0.5)
WEIGHT_LENGTH = 0.3     # 线长度权重 (v5优化: 0.3)
WEIGHT_ANGLE = 0.2      # 角度质量权重
