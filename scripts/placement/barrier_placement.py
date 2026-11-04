"""
路障等距铺设 — 透视感知版本
=============================

在道路图片中检测车道线后，沿每条车道线按真实世界等间距放置路障，
并自动匹配透视关系（近大远小）。

管道:
  1. 车道线检测 (Ultra-Fast-Lane-Detection)
  2. 估计地平线 / 消失点
  3. 透视感知等距采样（基于 1/(y - horizon) 世界深度模型）
  4. 从参考路障图提取路障（支持 alpha / GrabCut / 色键抠图）
  5. 带阴影和透视缩放的 alpha 合成
  6. 可选输出 ComfyUI 兼容掩膜（离散路障标记）
"""

import argparse
import os
import sys
import warnings
from pathlib import Path

import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image

warnings.filterwarnings("ignore", category=FutureWarning)

# 复用 detect_lanes 中的模型定义和工具函数
from detect_lanes import (
    CULANE_ROW_ANCHOR, GRIDING_NUM, CLS_NUM_PER_LANE, NUM_LANES,
    RESIZE_H, RESIZE_W, IMAGENET_MEAN, IMAGENET_STD, IMG_EXTENSIONS,
    ParsingNet, _init_weights, compute_lane_points,
    download_model, load_model,
)


# ============================================================================
# 透视感知车道线采样
# ============================================================================

def estimate_horizon_y(lane_points_list):
    """
    从多条车道线的远端（图像顶部）估计地平线 y 坐标。

    原理: 透视投影下平行车道线汇聚于消失点（地平线附近）。
    取每条车道线最远的几个点，用它们的 y 均值近似地平线高度。

    参数:
        lane_points_list: 列表的列表，每条车道线为 [(x, y), ...]

    返回:
        float: 估计的地平线 y 坐标，失败返回 None
    """
    far_points = []
    for pts in lane_points_list:
        pts_clean = [p for p in pts if p[0] > 0 and p[1] > 0]
        if len(pts_clean) < 3:
            continue
        # y 值最小的点 = 图像中最远的点
        pts_sorted = sorted(pts_clean, key=lambda p: p[1])
        far_points.extend(pts_sorted[:min(5, len(pts_sorted))])

    if not far_points:
        return None

    # 用较稳定的中位数而非均值，抗异常车道线干扰
    ys = [p[1] for p in far_points]
    return float(np.median(ys))


def sample_equal_distance_lane(lane_points, num_barriers, horizon_y,
                               head_blank_ratio=0.10, tail_blank_ratio=0.10):
    """
    沿一条车道线按真实世界等间距采样路障放置位置。

    透视模型:
      在平直道路的针孔相机透视下，实时世界中等距的点在图像中
      的 y 坐标满足:  世界深度 ∝ 1/(y - horizon_y)

    因此我们在"逆深度空间"中等距采样，再映射回图像坐标，
    即可保证路障在真实世界中均匀分布。

    参数:
        lane_points:      [(x, y), ...] 车道线点坐标（y 从小到大 ≈ 远→近）
        num_barriers:     需要放置的路障数量
        horizon_y:        估计的地平线 y 坐标
        head_blank_ratio: 车道线远端（靠近地平线）留空比例 (0-1)
        tail_blank_ratio: 车道线近端（靠近画面底部）留空比例 (0-1)

    返回:
        [(x, y, scale), ...] 每个路障的图像坐标和缩放比例
    """
    lane_points = [p for p in lane_points if p[0] > 0 and p[1] > 0]
    if len(lane_points) < 2:
        return []

    # 按 y 从小到大排序（远→近）
    lane_points = sorted(lane_points, key=lambda p: p[1])

    xs = np.array([p[0] for p in lane_points])
    ys = np.array([p[1] for p in lane_points])

    # 逆深度 = 1/(y - horizon)，代表该点在真实世界中的"深度"
    depths = 1.0 / (ys - horizon_y + 1e-6)

    # 相邻点之间的世界距离差 = 逆深度差的绝对值
    world_diffs = np.abs(np.diff(depths))
    # 累积世界距离
    cum_world = np.concatenate([[0], np.cumsum(world_diffs)])

    total_world = cum_world[-1]
    if total_world < 1e-8:
        return []

    # 首尾留空：裁剪采样范围
    head_blank = total_world * head_blank_ratio
    tail_blank = total_world * tail_blank_ratio
    start_dist = cum_world[0] + head_blank
    end_dist = cum_world[-1] - tail_blank

    if end_dist <= start_dist:
        return []  # 空白太大，车道线剩余部分不够放

    # 在裁剪后的世界距离区间内等距采样
    target_dists = np.linspace(start_dist, end_dist, num_barriers)

    # 基线缩放：以最近的车道点（y 最大）为 scale=1.0 的参考
    y_bottom = float(max(ys))
    depth_bottom = 1.0 / (y_bottom - horizon_y + 1e-6)

    results = []
    for target in target_dists:
        # 二分查找目标距离落在哪段区间
        idx = int(np.searchsorted(cum_world, target))
        idx = max(1, min(idx, len(lane_points) - 1))

        # 线性插值坐标
        seg_len = cum_world[idx] - cum_world[idx - 1]
        t = (target - cum_world[idx - 1]) / (seg_len + 1e-8)
        t = max(0.0, min(1.0, t))

        x = float(xs[idx - 1] + t * (xs[idx] - xs[idx - 1]))
        y = float(ys[idx - 1] + t * (ys[idx] - ys[idx - 1]))

        # 透视缩放: scale = depth_current / depth_bottom
        depth_current = 1.0 / (y - horizon_y + 1e-6)
        scale = float(depth_current / depth_bottom)

        # 防止极端情况
        scale = max(0.06, min(1.0, scale))

        results.append((int(x), int(y), scale))

    return results


# ============================================================================
# 车道线断点分析 + 路障放置策略
# ============================================================================

def classify_lane_segments(lane_pts, gray_img, edge_threshold=35,
                           gap_min_len=2, merge_gap_px=20):
    """
    分析一条车道线的连续性：区分连续段和断点缺口。

    原理: 沿车道线采样 Canny 边缘强度。
    高边缘强度 → 真实的白色/黄色标线 → "连续段"
    低边缘强度 → 标线缺失 → "缺口"

    参数:
        lane_pts:        [(x, y), ...] 车道线点坐标
        gray_img:        (H, W) 灰度原图
        edge_threshold:  边缘强度阈值 (0-255), 低于此值视为缺口
        gap_min_len:     至少几个连续点才构成一个有效缺口
        merge_gap_px:    两个缺口间距小于此值时合并为一个

    返回:
        (segments, gaps):
          segments — list of [(x,y),...] 连续段
          gaps     — list of [(x,y),...] 缺口段 (需要放置路障的位置)
    """
    if len(lane_pts) < 4:
        return [lane_pts], []

    # ---- Canny 边缘图 ----
    edges = cv2.Canny(gray_img, 40, 140)
    h, w = edges.shape

    # ---- 沿车道线采样边缘强度 ----
    strengths = []
    for px, py in lane_pts:
        px, py = int(px), int(py)
        x1 = max(0, px - 3)
        x2 = min(w, px + 4)
        y1 = max(0, py - 3)
        y2 = min(h, py + 4)
        patch = edges[y1:y2, x1:x2]
        strengths.append(float(np.mean(patch)))

    strengths = np.array(strengths)

    # ---- 标记连续/缺口 ----
    is_edge = strengths >= edge_threshold

    # ---- 形态学平滑：单点噪声不影响分类 ----
    smoothed = is_edge.copy()
    for i in range(1, len(is_edge) - 1):
        if is_edge[i - 1] and is_edge[i + 1]:
            smoothed[i] = True
        elif not is_edge[i - 1] and not is_edge[i + 1]:
            smoothed[i] = False

    # ---- 提取连续段和缺口段 ----
    segments, gaps = [], []
    i = 0
    while i < len(smoothed):
        if smoothed[i]:
            seg = []
            while i < len(smoothed) and smoothed[i]:
                seg.append(lane_pts[i])
                i += 1
            if len(seg) >= gap_min_len:
                segments.append(seg)
        else:
            gap = []
            while i < len(smoothed) and not smoothed[i]:
                gap.append(lane_pts[i])
                i += 1
            if len(gap) >= gap_min_len:
                gaps.append(gap)

    # ---- 合并间距太近的缺口 ----
    if len(gaps) > 1 and merge_gap_px > 0:
        merged = []
        current_gap = gaps[0]
        for next_gap in gaps[1:]:
            # 两个缺口之间的 y 距离
            dy = abs(next_gap[0][1] - current_gap[-1][1])
            if dy < merge_gap_px:
                current_gap.extend(next_gap)
            else:
                merged.append(current_gap)
                current_gap = next_gap
        merged.append(current_gap)
        gaps = merged

    return segments, gaps


# ============================================================================
# 路障提取
# ============================================================================

def extract_barrier_from_image(barrier_image, method='auto'):
    """
    从参考路障图中提取路障，返回带透明通道的 RGBA 图像。

    参数:
        barrier_image:  (H, W, 3) BGR 或 (H, W, 4) BGRA 数组
        method:
          'auto'         — 有 alpha 则用 alpha，否则尝试 GrabCut
          'alpha'        — 直接用 alpha 通道（图片需自带透明度）
          'green_screen' — 绿幕抠图（绿色背景→透明）
          'white_bg'     — 白色背景移除
          'grabcut'      — 强制 GrabCut 前景分割

    返回:
        (H, W, 4) BGRA uint8 数组，背景透明
    """
    h, w = barrier_image.shape[:2]

    # ---- 自动判断 ----
    if method == 'auto':
        if barrier_image.shape[2] == 4:
            alpha_channel = barrier_image[:, :, 3]
            if np.any(alpha_channel < 250):  # 确实有透明区域
                return barrier_image.copy()
        method = 'grabcut'  # 回退

    # ---- 直接用 alpha ----
    if method == 'alpha' and barrier_image.shape[2] == 4:
        return barrier_image.copy()

    # ---- 转 RGBA ----
    if barrier_image.shape[2] == 3:
        rgba = cv2.cvtColor(barrier_image, cv2.COLOR_BGR2BGRA)
    else:
        rgba = barrier_image.copy()

    # ---- 绿幕抠图 ----
    if method == 'green_screen':
        # 绿色范围（宽松）
        lower_green = np.array([0, 80, 0])
        upper_green = np.array([80, 255, 80])
        green_mask = cv2.inRange(rgba[:, :, :3], lower_green, upper_green)

        # 形态学处理去掉噪点
        kernel = np.ones((3, 3), np.uint8)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        green_mask = cv2.morphologyEx(green_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        rgba[:, :, 3] = 255 - green_mask
        return rgba

    # ---- 白色背景移除 ----
    if method == 'white_bg':
        gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2GRAY)
        _, white_mask = cv2.threshold(gray, 240, 255, cv2.THRESH_BINARY)
        rgba[:, :, 3] = 255 - white_mask
        return rgba

    # ---- GrabCut 前景分割 ----
    if method == 'grabcut':
        gray = cv2.cvtColor(rgba[:, :, :3], cv2.COLOR_BGR2GRAY)

        # Canny 边缘检测 → 找到物体大致轮廓
        edges = cv2.Canny(gray, 30, 120)
        kernel = np.ones((7, 7), np.uint8)
        dilated = cv2.dilate(edges, kernel, iterations=3)

        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if not contours:
            # 找不到轮廓 → 整张图作为物体
            rgba[:, :, 3] = 255
            return rgba

        # 取最大轮廓的边界框
        largest = max(contours, key=cv2.contourArea)
        x_bb, y_bb, bw, bh = cv2.boundingRect(largest)

        # 扩展边距
        pad = 15
        x_bb = max(0, x_bb - pad)
        y_bb = max(0, y_bb - pad)
        bw = min(w - x_bb, bw + 2 * pad)
        bh = min(h - y_bb, bh + 2 * pad)

        if bw < 5 or bh < 5:
            rgba[:, :, 3] = 255
            return rgba

        # 初始化 GrabCut
        mask = np.zeros((h, w), np.uint8)
        # 边界框内标记为"可能前景"
        mask[y_bb:y_bb + bh, x_bb:x_bb + bw] = cv2.GC_PR_FGD
        # 边界框外标记为"确定背景"
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        rect = (x_bb, y_bb, bw, bh)

        try:
            cv2.grabCut(rgba[:, :, :3], mask, rect,
                        bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_MASK)
        except cv2.error:
            # 回退：前景=边界框内
            simple_mask = np.zeros((h, w), dtype=np.uint8)
            simple_mask[y_bb:y_bb + bh, x_bb:x_bb + bw] = 255
            rgba[:, :, 3] = simple_mask
            return rgba

        # 保留"确定前景"和"可能前景"区域
        final_mask = np.where(
            (mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD), 255, 0
        ).astype(np.uint8)

        # 只保留最大连通区域（去除 GrabCut 的碎片）
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            final_mask, connectivity=8
        )
        if num_labels > 1:
            # 跳过背景 label=0，找最大的前景区域
            largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
            final_mask[labels != largest_label] = 0

        rgba[:, :, 3] = final_mask
        return rgba

    # 未知方法，返回原图
    return rgba


# ============================================================================
# 合成与透视渲染
# ============================================================================

def add_shadow(background, x, y, scale, shadow_alpha=0.35):
    """
    在路障底部绘制椭圆阴影，增强真实感。

    阴影大小和透明度都随透视缩放调整。
    """
    bg_h, bg_w = background.shape[:2]
    shadow_w = max(2, int(40 * scale))
    shadow_h = max(1, int(12 * scale))

    if x < 0 or x >= bg_w or y < 0 or y >= bg_h:
        return background

    # 绘制椭圆阴影
    shadow = np.zeros((shadow_h * 2 + 4, shadow_w * 2 + 4), dtype=np.float32)
    cx = shadow_w + 2
    cy = shadow_h + 2
    cv2.ellipse(shadow, (cx, cy), (shadow_w, shadow_h),
                0, 0, 360, 1.0, -1)

    # 高斯模糊软化边缘，越远的路障模糊越强
    blur_ksize = max(3, shadow_w // 3) | 1  # 保证奇数
    shadow = cv2.GaussianBlur(shadow, (blur_ksize, blur_ksize), 0)

    # 按缩放进一步减小远处阴影的不透明度
    effective_alpha = shadow_alpha * (0.5 + 0.5 * scale)

    # 定位
    px = x - cx
    py = y - shadow_h  # 底部≈采样点
    sx, sy = px, py
    sh, sw = shadow.shape

    # 裁剪在图像范围内的区域
    src_x1 = max(0, -sx)
    src_y1 = max(0, -sy)
    src_x2 = min(sw, bg_w - sx)
    src_y2 = min(sh, bg_h - sy)
    dst_x1 = max(0, sx)
    dst_y1 = max(0, sy)
    dst_x2 = min(bg_w, sx + sw)
    dst_y2 = min(bg_h, sy + sh)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return background

    shadow_crop = shadow[src_y1:src_y2, src_x1:src_x2]
    roi = background[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)

    # Alpha 合成（阴影是变暗而不是叠加）
    alpha = shadow_crop * effective_alpha
    roi[:, :, 0] = roi[:, :, 0] * (1 - alpha)
    roi[:, :, 1] = roi[:, :, 1] * (1 - alpha)
    roi[:, :, 2] = roi[:, :, 2] * (1 - alpha)

    background[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(roi, 0, 255).astype(np.uint8)
    return background


def place_barrier(background, barrier_rgba, x, y, scale):
    """
    将路障合成到背景图上指定位置。

    参数:
        background:   (H, W, 3) BGR 背景图
        barrier_rgba: (BH, BW, 4) BGRA 路障图
        x, y:         放置位置 — 路障底部中心
        scale:        float, 透视缩放比例
    """
    bh, bw = barrier_rgba.shape[:2]
    bg_h, bg_w = background.shape[:2]

    # 缩放
    new_w = max(1, int(bw * scale))
    new_h = max(1, int(bh * scale))
    if new_w < 2 or new_h < 2:
        return background  # 太小，不画

    resized = cv2.resize(barrier_rgba, (new_w, new_h), interpolation=cv2.INTER_AREA)

    # 计算粘贴区域: 路障底部中心对齐到 (x, y)
    paste_x = x - new_w // 2
    paste_y = y - new_h

    # 裁剪在图像范围内的区域
    src_x1 = max(0, -paste_x)
    src_y1 = max(0, -paste_y)
    src_x2 = min(new_w, bg_w - paste_x)
    src_y2 = min(new_h, bg_h - paste_y)
    dst_x1 = max(0, paste_x)
    dst_y1 = max(0, paste_y)
    dst_x2 = min(bg_w, paste_x + new_w)
    dst_y2 = min(bg_h, paste_y + new_h)

    if src_x2 <= src_x1 or src_y2 <= src_y1:
        return background

    roi = background[dst_y1:dst_y2, dst_x1:dst_x2].astype(np.float32)
    overlay = resized[src_y1:src_y2, src_x1:src_x2].astype(np.float32)

    # Alpha 合成
    alpha = overlay[:, :, 3:4] / 255.0
    blended = overlay[:, :, :3] * alpha + roi * (1.0 - alpha)
    background[dst_y1:dst_y2, dst_x1:dst_x2] = np.clip(blended, 0, 255).astype(np.uint8)

    return background


# ============================================================================
# 掩膜输出（供 ComfyUI 使用）
# ============================================================================

def generate_lane_mask(lanes, h, w, lane_width=25):
    """
    生成车道线掩膜 — 纯绿色线条, 供 ComfyUI 做 lane inpainting。

    G 通道 = 255, ComfyUI ImageToMask(channel="green") 提取。

    参数:
        lanes:      [[(x,y),...], ...]  车道线点坐标
        h, w:      图片尺寸
        lane_width: 线条粗细
    """
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for pts in lanes:
        pts_clean = [p for p in pts if p[0] > 0]
        if len(pts_clean) >= 2:
            pts_array = np.array(pts_clean, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(mask, [pts_array], isClosed=False,
                          color=(0, 255, 0), thickness=lane_width)
    return mask


def generate_barrier_positions_mask(barrier_positions, h, w,
                                     barrier_base_w=40, barrier_base_h=100):
    """
    生成路障位置掩膜 — 红色椭圆 + 白色十字, 供 ComfyUI 做 barrier inpainting。

    BGR = (0, 0, 255) 即纯红色, PNG 保存为红色。
    ComfyUI 读取后 RGB = (255, 0, 0), ImageToMask(channel="red") 提取。

    参数:
        barrier_positions: [(x, y, scale), ...]  路障放置位置
        h, w:              图片尺寸
        barrier_base_w/h:  路障标记基础尺寸（近端时）
    """
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    for px, py, scale in barrier_positions:
        ew = max(30, int(barrier_base_w * scale))
        eh = max(60, int(barrier_base_h * scale))
        cx, cy = px, py - eh // 3

        # 红色填充椭圆
        cv2.ellipse(mask, (cx, cy), (ew, eh), 0, 0, 360,
                    (0, 0, 255), -1)

        # 白色十字准星
        cross_len = max(10, int(eh * 0.45))
        cv2.line(mask, (cx, cy - cross_len), (cx, cy + cross_len),
                 (255, 255, 255), max(3, int(5 * scale)))
        cv2.line(mask, (cx - cross_len, cy), (cx + cross_len, cy),
                 (255, 255, 255), max(3, int(5 * scale)))

    return mask


def generate_single_barrier_mask(barrier_position, h, w,
                                  barrier_base_w=80, barrier_base_h=200):
    """
    生成只含一个路障的独立掩膜 — 红色椭圆 + 白色十字。

    与 generate_barrier_positions_mask 的区别: 只画一个路障标记,
    供 ComfyUI 逐个生成路障时使用。

    参数:
        barrier_position: single (x, y, scale) tuple
        h, w:             图片尺寸
        barrier_base_w/h: 路障标记基础尺寸
    """
    mask = np.zeros((h, w, 3), dtype=np.uint8)
    px, py, scale = barrier_position
    ew = max(30, int(barrier_base_w * scale))
    eh = max(60, int(barrier_base_h * scale))
    cx, cy = px, py - eh // 3

    cv2.ellipse(mask, (cx, cy), (ew, eh), 0, 0, 360,
                (0, 0, 255), -1)                   # 红色填充
    cross_len = max(10, int(eh * 0.45))
    cv2.line(mask, (cx, cy - cross_len), (cx, cy + cross_len),
             (255, 255, 255), max(3, int(5 * scale)))
    cv2.line(mask, (cx - cross_len, cy), (cx + cross_len, cy),
             (255, 255, 255), max(3, int(5 * scale)))

    return mask


def generate_combined_mask(lanes, barrier_positions, h, w,
                           lane_width=25, barrier_base_w=40, barrier_base_h=100):
    """
    生成合并掩膜（车道线 + 路障在同一张图上），兼容旧的 --output-mask 行为。

    车道线 = 绿色 (BGR: 0,255,0), 路障 = 红色 (BGR: 0,0,255)。
    ComfyUI 中车道线用 green 通道、路障用 red 通道分别提取。

    参数:
        lanes:             [[(x,y),...], ...]  车道线点坐标
        barrier_positions: [(x, y, scale), ...]  路障位置
        h, w:             图片尺寸
    """
    mask = generate_lane_mask(lanes, h, w, lane_width=lane_width)
    for px, py, scale in barrier_positions:
        ew = max(30, int(barrier_base_w * scale))
        eh = max(60, int(barrier_base_h * scale))
        cx, cy = px, py - eh // 3
        cv2.ellipse(mask, (cx, cy), (ew, eh), 0, 0, 360,
                    (0, 0, 255), -1)  # 红色
        cross_len = max(10, int(eh * 0.45))
        cv2.line(mask, (cx, cy - cross_len), (cx, cy + cross_len),
                 (255, 255, 255), max(3, int(5 * scale)))
        cv2.line(mask, (cx - cross_len, cy), (cx + cross_len, cy),
                 (255, 255, 255), max(3, int(5 * scale)))
    return mask


# ============================================================================
# 主流程
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="沿车道线等距铺设路障 — 透视感知 + 断点检测版本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 基础使用：整条车道线等距放路障，输出两张独立掩膜
  python barrier_placement.py --input-image ./input/road.jpg \\
      --barrier-image ./barrier.png --output-image ./result.jpg \\
      --output-lane-mask ./lane_mask.png \\
      --output-barrier-mask ./barrier_mask.png

  # 只输出掩膜，不做图像合成（跳过路障抠图）
  python barrier_placement.py --input-image ./input/road.jpg \\
      --barrier-image ./barrier.png --output-image ./dummy.jpg \\
      --output-lane-mask ./lane_mask.png \\
      --output-barrier-mask ./barrier_mask.png

  # 调整首尾留空
  python barrier_placement.py --input-image ./input/road.jpg \\
      --barrier-image ./barrier.png --output-image ./result.jpg \\
      --output-lane-mask ./lane_mask.png \\
      --output-barrier-mask ./barrier_mask.png \\
      --head-blank-ratio 0.15 --tail-blank-ratio 0.05
        """
    )

    # 必需参数
    parser.add_argument('--input-image', required=True,
                        help='输入道路图片路径')
    parser.add_argument('--barrier-image', required=True,
                        help='路障参考图片路径')
    parser.add_argument('--output-image', default='./output_with_barriers.jpg',
                        help='输出合成图片路径')

    # 车道线检测参数
    parser.add_argument('--model-path', default='./culane_18.pth',
                        help='车道线检测模型路径')
    parser.add_argument('--lane-width', type=int, default=25,
                        help='车道线绘制宽度（掩膜用）')
    parser.add_argument('--barrier-marker-w', type=int, default=80,
                        help='路障掩膜标记基础宽度 px（近端）')
    parser.add_argument('--barrier-marker-h', type=int, default=200,
                        help='路障掩膜标记基础高度 px')

    # --- 路障放置策略 ---
    parser.add_argument('--num-barriers', type=int, default=2,
                        help='每条车道线放置的路障数量')
    parser.add_argument('--barrier-on-segments', default='all',
                        choices=['gaps', 'all', 'continuous'],
                        help='在哪些车道段放置路障')
    parser.add_argument('--head-blank-ratio', type=float, default=0.10,
                        help='车道线远端留空比例')
    parser.add_argument('--tail-blank-ratio', type=float, default=0.10,
                        help='车道线近端留空比例')
    parser.add_argument('--barrier-on-lanes', default=None,
                        help='只处理指定车道线索引，逗号分隔')
    parser.add_argument('--horizon-y', type=float, default=None,
                        help='手动指定地平线 y 坐标')
    parser.add_argument('--gap-edge-threshold', type=int, default=35,
                        help='缺口边缘强度阈值')
    parser.add_argument('--gap-min-length', type=int, default=2,
                        help='至少多少像素才构成有效缺口')
    parser.add_argument('--barrier-extract-method', default='auto',
                        choices=['auto', 'alpha', 'green_screen', 'white_bg', 'grabcut'],
                        help='路障抠图方法')
    parser.add_argument('--barrier-scale-bias', type=float, default=0.8,
                        help='路障整体缩放偏置')
    parser.add_argument('--output-lane-mask', default=None,
                        help='车道线掩膜路径')
    parser.add_argument('--output-barrier-mask', default=None,
                        help='路障掩膜路径（合并）')
    parser.add_argument('--output-individual-barrier-masks', default=None,
                        help='独立路障掩膜路径前缀, 如 ./masks/barrier')
    parser.add_argument('--output-mask', default=None,
                        help='合并掩膜路径（兼容旧用法）')
    parser.add_argument('--no-shadow', action='store_true',
                        help='不添加路障阴影')
    parser.add_argument('--no-cuda', action='store_true',
                        help='强制使用 CPU')

    args = parser.parse_args()

    # ------------------------------------------------------------------
    # 1. 设备与模型
    # ------------------------------------------------------------------
    device = torch.device('cuda' if torch.cuda.is_available() and not args.no_cuda else 'cpu')
    print(f"[设备] {device}")

    model_path = Path(args.model_path)
    if not model_path.exists():
        if not download_model(model_path):
            sys.exit(1)
    model = load_model(model_path, device)

    # ------------------------------------------------------------------
    # 2. 加载原图
    # ------------------------------------------------------------------
    print(f"[加载] 原图: {args.input_image}")
    img = cv2.imread(args.input_image)
    if img is None:
        print("错误: 无法读取输入图片")
        sys.exit(1)

    orig_h, orig_w = img.shape[:2]
    print(f"[图片] 尺寸: {orig_w} x {orig_h}")

    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    transform = transforms.Compose([
        transforms.Resize((RESIZE_H, RESIZE_W)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_pil = Image.fromarray(img_rgb)
    tensor = transform(img_pil).unsqueeze(0).to(device)

    # ------------------------------------------------------------------
    # 3. 车道线检测 (UFLD)
    # ------------------------------------------------------------------
    print("[检测] 车道线 (UFLD 深度学习) ...")
    with torch.no_grad():
        output = model(tensor)

    lanes = compute_lane_points(output, orig_w, orig_h)

    all_lanes_raw = []
    for pts in lanes:
        pts_clean = [tuple(p) for p in pts if p[0] > 0 and p[1] > 0]
        if len(pts_clean) >= 3:
            all_lanes_raw.append(pts_clean)

    if not all_lanes_raw:
        print("[警告] 未检测到有效车道线")
        _write_masks(args, [], [], orig_h, orig_w)
        cv2.imwrite(args.output_image, img)
        print(f"[完成] 输出: {args.output_image}")
        return

    print(f"[检测] 有效车道线: {len(all_lanes_raw)} 条")

    # ------------------------------------------------------------------
    # 4. 断点分析
    # ------------------------------------------------------------------
    all_continuous_segments = []
    all_gap_segments = []
    print(f"[分析] 车道线连续性 ...")
    for i, lane_pts in enumerate(all_lanes_raw):
        segs, gaps = classify_lane_segments(
            lane_pts, gray_img,
            edge_threshold=args.gap_edge_threshold,
            gap_min_len=args.gap_min_length,
        )
        all_continuous_segments.extend(segs)
        all_gap_segments.extend(gaps)
        print(f"  [车道线 {i}] 连续段: {len(segs)}, 缺口段: {len(gaps)}")

    # ------------------------------------------------------------------
    # 5. 路障放置策略
    # ------------------------------------------------------------------
    if args.barrier_on_segments == 'all':
        target_segments = all_lanes_raw
        mode = "整条车道线"
    elif args.barrier_on_segments == 'gaps':
        target_segments = all_gap_segments
        mode = "缺口段"
    elif args.barrier_on_segments == 'continuous':
        target_segments = all_continuous_segments
        mode = "连续段"
    else:
        target_segments = all_gap_segments + all_continuous_segments
        mode = "全部段"
    print(f"[策略] {mode}放置, 共 {len(target_segments)} 段")

    # 车道线索引过滤
    if args.barrier_on_lanes is not None:
        lane_indices = [int(x.strip()) for x in args.barrier_on_lanes.split(',')]
        if args.barrier_on_segments == 'all':
            target_segments = [all_lanes_raw[idx] for idx in lane_indices if 0 <= idx < len(all_lanes_raw)]
        print(f"[过滤] 车道线索引 {lane_indices}, 剩余 {len(target_segments)} 条")

    # ------------------------------------------------------------------
    # 6. 地平线 + 透视采样
    # ------------------------------------------------------------------
    horizon_y = args.horizon_y if args.horizon_y is not None else estimate_horizon_y(all_lanes_raw)
    if horizon_y is None:
        horizon_y = float(min(p[1] for lane in all_lanes_raw for p in lane))
    print(f"[地平线] y = {horizon_y:.1f}")

    all_barrier_positions = []
    for seg_idx, seg_pts in enumerate(target_segments):
        seg_sorted = sorted(seg_pts, key=lambda p: p[1])
        min_y = horizon_y + (orig_h - horizon_y) * 0.03
        seg_sorted = [p for p in seg_sorted if p[1] > min_y]
        if len(seg_sorted) < 2:
            continue

        seg_len = abs(seg_sorted[-1][1] - seg_sorted[0][1])
        adaptive_num = args.num_barriers
        if seg_len < 20:
            adaptive_num = 1
        elif seg_len < 60:
            adaptive_num = max(1, args.num_barriers // 2)

        if args.barrier_on_segments == 'all':
            positions = sample_equal_distance_lane(
                seg_sorted, adaptive_num, horizon_y,
                head_blank_ratio=args.head_blank_ratio,
                tail_blank_ratio=args.tail_blank_ratio,
            )
        else:
            positions = sample_equal_distance_lane(
                seg_sorted, adaptive_num, horizon_y,
                head_blank_ratio=0.0, tail_blank_ratio=0.0,
            )

        for j in range(len(positions)):
            px, py, sc = positions[j]
            positions[j] = (px, py, sc * args.barrier_scale_bias)

        print(f"  [段 {seg_idx}] y={seg_sorted[0][1]:.0f}~{seg_sorted[-1][1]:.0f} → {len(positions)} 个路障")
        all_barrier_positions.extend(positions)

    if not all_barrier_positions:
        print("[警告] 未能计算出有效路障位置")
        _write_masks(args, all_lanes_raw, [], orig_h, orig_w)
        cv2.imwrite(args.output_image, img)
        print(f"[完成] 输出: {args.output_image}")
        return

    print(f"[路障] 共计 {len(all_barrier_positions)} 个放置位置")

    # ------------------------------------------------------------------
    # 7. 合成（路障图片可选）
    # ------------------------------------------------------------------
    result = img.copy()
    barrier_loaded = False
    if args.barrier_image and os.path.exists(args.barrier_image):
        print(f"[加载] 路障图: {args.barrier_image}")
        barrier_img = cv2.imread(args.barrier_image, cv2.IMREAD_UNCHANGED)
        if barrier_img is not None:
            barrier_rgba = extract_barrier_from_image(barrier_img, method=args.barrier_extract_method)
            alpha_valid = np.count_nonzero(barrier_rgba[:, :, 3])
            alpha_ratio = alpha_valid / max(1, barrier_rgba.shape[0] * barrier_rgba.shape[1])
            print(f"[抠图] 前景像素比: {alpha_ratio:.1%}")
            barrier_loaded = True
            all_barrier_positions.sort(key=lambda p: p[1])
            for px, py, scale in all_barrier_positions:
                if not args.no_shadow:
                    result = add_shadow(result, px, py, scale)
                result = place_barrier(result, barrier_rgba, px, py, scale)

    if not barrier_loaded:
        print("[提示] 未加载路障参考图, 仅输出掩膜")

    cv2.imwrite(args.output_image, result, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"[输出] 合成图: {args.output_image}")

    # ------------------------------------------------------------------
    # 8. 输出掩膜
    # ------------------------------------------------------------------
    _write_masks(args, all_lanes_raw, all_barrier_positions, orig_h, orig_w)

    print("[完成]")


def _write_masks(args, lanes, barriers, h, w):
    """输出各类型掩膜图片"""
    if args.output_lane_mask:
        mask = generate_lane_mask(lanes, h, w, lane_width=args.lane_width)
        cv2.imwrite(args.output_lane_mask, mask)
        print(f"[输出] 车道线掩膜: {args.output_lane_mask}")

    if args.output_barrier_mask:
        mask = generate_barrier_positions_mask(
            barriers, h, w,
            barrier_base_w=args.barrier_marker_w,
            barrier_base_h=args.barrier_marker_h,
        )
        cv2.imwrite(args.output_barrier_mask, mask)
        print(f"[输出] 路障掩膜(合并): {args.output_barrier_mask}")

    if args.output_individual_barrier_masks and barriers:
        base = args.output_individual_barrier_masks.rstrip('/\\')
        parent = os.path.dirname(base)
        if parent:
            os.makedirs(parent, exist_ok=True)
        for i, pos in enumerate(barriers):
            single = generate_single_barrier_mask(
                pos, h, w,
                barrier_base_w=args.barrier_marker_w,
                barrier_base_h=args.barrier_marker_h,
            )
            path = f"{base}_{i:03d}.png"
            cv2.imwrite(path, single)
        print(f"[输出] 独立路障掩膜 x{len(barriers)}: {base}_000.png ~ {base}_{len(barriers)-1:03d}.png")

    if args.output_mask:
        combined = generate_combined_mask(
            lanes, barriers, h, w,
            lane_width=args.lane_width,
            barrier_base_w=args.barrier_marker_w,
            barrier_base_h=args.barrier_marker_h,
        )
        cv2.imwrite(args.output_mask, combined)
        print(f"[输出] 合并掩膜: {args.output_mask}")


if __name__ == '__main__':
    main()
