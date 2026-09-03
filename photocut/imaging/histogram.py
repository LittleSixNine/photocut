# -*- coding: utf-8 -*-
"""
老照片处理算法 - 推荐 PS 色阶模式

三种模式:
1. ps_levels() / auto_enhance() - PS 色阶模式（推荐！简单有效）
2. histogram_stretch()           - 百分位模式（自适应，通用对比度增强）
3. white_balance_yellow()        - 白平衡去黄（HSV双通道调整）

推荐使用（简单有效）:
    from photocut.imaging.histogram import ps_levels, auto_enhance
    # 默认参数: highlight=200，效果最佳
    result = ps_levels(img)  # 或 auto_enhance(img)
"""

import cv2
import numpy as np


def histogram_stretch(channel, low_percentile=5, high_percentile=95):
    """
    直方图拉伸：扩展对比度以覆盖发黄的白边（百分位模式 - 自适应）

    参数:
        channel: 单通道图像（通常是亮度通道）
        low_percentile: 低百分位阈值（默认 5）
        high_percentile: 高百分位阈值（默认 95）

    返回:
        stretched: 拉伸后的通道

    示例:
        # 对 HSV 的 V 通道进行拉伸
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = histogram_stretch(v)
        hsv_stretched = cv2.merge([h, s, v])
        img_stretched = cv2.cvtColor(hsv_stretched, cv2.COLOR_HSV2BGR)
    """
    # 使用 numpy percentile 计算百分位数（更简洁高效）
    low_thresh, high_thresh = np.percentile(channel, [low_percentile, high_percentile])

    # 避免除零
    if high_thresh <= low_thresh:
        return channel

    # 拉伸到 0-255
    stretched = np.zeros_like(channel, dtype=np.float32)
    mask = (channel >= low_thresh) & (channel <= high_thresh)
    stretched[mask] = (channel[mask] - low_thresh) / (high_thresh - low_thresh) * 255
    stretched[channel < low_thresh] = 0
    stretched[channel > high_thresh] = 255

    return stretched.astype(np.uint8)


def histogram_stretch_fixed(channel, low_threshold=10, high_threshold=200):
    """
    直方图拉伸：固定阈值模式
    亮度高于 high_threshold 的全部变成 255（纯白）
    亮度低于 low_threshold 的全部变成 0（纯黑）

    参数:
        channel: 单通道图像（通常是亮度通道）
        low_threshold: 暗部固定阈值（默认 10）
        high_threshold: 亮部固定阈值（默认 200）
                         高于此值的全部变为 255

    返回:
        stretched: 拉伸后的通道

    示例:
        # 让亮度高于 200 的全部变成白色
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h, s, v = cv2.split(hsv)
        v = histogram_stretch_fixed(v, high_threshold=200)
        hsv_stretched = cv2.merge([h, s, v])
        img_stretched = cv2.cvtColor(hsv_stretched, cv2.COLOR_HSV2BGR)
    """
    # 创建输出数组
    stretched = np.zeros_like(channel, dtype=np.float32)

    # 中间区域线性拉伸
    mask = (channel >= low_threshold) & (channel <= high_threshold)
    stretched[mask] = (channel[mask] - low_threshold) / (high_threshold - low_threshold) * 255

    # 极端值截断
    stretched[channel < low_threshold] = 0
    stretched[channel > high_threshold] = 255

    return stretched.astype(np.uint8)


def white_balance_yellow(img, v_threshold=200, s_threshold=80):
    """
    白平衡去黄：把发黄的亮部区域变成纯白

    原理：
    - 发黄区域特征：V > v_threshold（较亮），但 S > 0（有颜色饱和度）
    - 处理：把符合条件的区域 S 设为 0（去色），V 设为 255（最亮）

    参数:
        img: BGR 图像
        v_threshold: 亮度阈值，高于此值的区域处理（默认 200）
        s_threshold: 饱和度阈值，低于此值的区域才去饱和（默认 80）
                     避免把鲜艳的黄色物体也变成白色

    返回:
        result: 处理后的 BGR 图像

    示例:
        # 把亮度>200 的发黄区域变成白色
        result = white_balance_yellow(img, v_threshold=200)
    """
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    h, s, v = cv2.split(hsv)

    # 创建掩码：亮度高但还有饱和度的区域（发黄的白边）
    # V > v_threshold: 较亮的区域
    # S < s_threshold: 不是特别鲜艳的颜色（避免把黄衣服、黄花变成白色）
    yellow_white_mask = (v > v_threshold) & (s < s_threshold)

    # 处理：去饱和 + 提亮
    s_processed = s.copy()
    v_processed = v.copy()

    s_processed[yellow_white_mask] = 0  # 去饱和 -> 变成灰/白
    v_processed[yellow_white_mask] = 255  # 最亮 -> 变成纯白

    # 合并回 HSV
    hsv_processed = cv2.merge([h, s_processed, v_processed])
    result = cv2.cvtColor(hsv_processed, cv2.COLOR_HSV2BGR)

    return result


def levels_rgb(img, shadow=0, gamma=1.0, highlight=200):
    """
    模拟 PS 色阶调整（RGB 模式）

    原理：
    - 在 RGB 空间对 R、G、B 三个通道应用相同的映射
    - 把输入范围 [shadow, highlight] 映射到输出 [0, 255]
    - highlight 越小，亮部拉伸越激进

    参数:
        img: BGR 图像
        shadow: 黑点（输入阴影），默认 0
        gamma: 伽马值，默认 1.0（线性）
        highlight: 白点（输入高光），默认 200
                   PS 中的滑块：255 -> 200 就是这个参数

    返回:
        result: 处理后的 BGR 图像

    示例:
        # 模拟 PS：把白点从 255 拖到 200
        result = levels_rgb(img, shadow=0, gamma=1.0, highlight=200)
    """
    # 转换为 float 进行计算
    img_float = img.astype(np.float32)

    # 创建查找表（LUT）用于加速
    lut = np.zeros(256, dtype=np.uint8)

    for i in range(256):
        if i <= shadow:
            lut[i] = 0
        elif i >= highlight:
            lut[i] = 255
        else:
            # 线性映射 + gamma 调整
            normalized = (i - shadow) / (highlight - shadow)
            if gamma != 1.0:
                normalized = np.power(normalized, gamma)
            lut[i] = int(normalized * 255)

    # 应用 LUT 到每个通道
    result = cv2.LUT(img, lut)

    return result


def ps_levels(img, highlight=200, shadow=0, gamma=1.0):
    """
    PS 色阶调整 - 推荐算法！

    模拟 Photoshop 色阶调整，在 RGB 空间对三个通道应用相同的映射。
    简单有效，是老照片去黄的最佳选择。

    重要说明：
    - 本函数返回新的图像数组，不会修改输入的原图
    - 仅用于角点检测辅助，不应作为最终裁剪的图像

    参数:
        img: BGR 图像 (numpy array)
        highlight: 白点阈值，默认 200（推荐值）
                   越低 = 越激进（更多区域变白）
                   越高 = 越保守（接近原图）
        shadow: 黑点阈值，默认 0
        gamma: 伽马值，默认 1.0（线性映射）

    返回:
        result: 处理后的 BGR 图像（新数组，不是原图引用）

    使用示例:
        from photocut.imaging.histogram import ps_levels

        # 默认参数，效果最好
        result = ps_levels(img)

        # 自定义参数
        result = ps_levels(img, highlight=190)  # 更激进
        result = ps_levels(img, highlight=220)  # 更保守
    """
    return levels_rgb(img, shadow=shadow, gamma=gamma, highlight=highlight)


# 别名：更方便的函数名
auto_enhance = ps_levels


if __name__ == "__main__":
    # 测试代码
    import sys
    import os

    if len(sys.argv) > 1:
        img_path = sys.argv[1]
        img = cv2.imread(img_path)
        if img is not None:
            h, w = img.shape[:2]
            # 缩放以便显示
            scale = 800 / max(h, w)
            display = cv2.resize(img, (int(w*scale), int(h*scale)))

            # 原图
            cv2.imshow("Original", display)

            # 拉伸后的亮度
            hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            v_stretched = histogram_stretch(v)
            hsv_stretched = cv2.merge([h, s, v_stretched])
            img_stretched = cv2.cvtColor(hsv_stretched, cv2.COLOR_HSV2BGR)
            display_stretched = cv2.resize(img_stretched, (int(w*scale), int(h*scale)))
            cv2.imshow("Stretched", display_stretched)

            # 白色掩码对比
            mask_orig = np.logical_and(s < 30, v > 200).astype(np.uint8) * 255
            mask_stretched = np.logical_and(s < 30, v_stretched > 200).astype(np.uint8) * 255
            cv2.imshow("Mask Original", cv2.resize(mask_orig, (int(w*scale), int(h*scale))))
            cv2.imshow("Mask Stretched", cv2.resize(mask_stretched, (int(w*scale), int(h*scale))))

            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print(f"无法读取图片: {img_path}")
    else:
        print("用法: python histogram_stretch.py <图片路径>")
