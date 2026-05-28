"""
基于传统图像处理技术的图像合成系统
Image Composition System (Traditional Methods)

作者: 张三
课程: 数字图像处理
学校: 南阳理工学院·计算机与软件学院

功能概述:
    1. 图像预处理 (灰度化、直方图均衡化、去噪)
    2. 前景提取 (GrabCut、颜色阈值、Canny+形态学三种方式)
    3. 图像合成 (加权平均、多频带融合、颜色融合、泊松融合)
    4. 后处理增强 (对比度、锐化、色彩调整)
    5. 图形界面 (Tkinter，支持参数调节与结果展示)

依赖安装:
    pip install opencv-python numpy pillow

运行:
    python image_composite_system.py
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os


# ============================================================
# 一、图像预处理模块
# ============================================================

class Preprocessor:
    """
    对输入的前景图像和背景图像进行预处理。
    预处理目的：统一图像尺寸、去噪、增强对比度，为后续合成提供质量稳定的输入。
    """

    @staticmethod
    def resize_to_match(fg, bg):
        """
        将前景图像缩放到不超过背景尺寸的合理范围，
        并保持原始宽高比，以便嵌入背景。

        参数:
            fg (np.ndarray): 前景图像 (BGR)
            bg (np.ndarray): 背景图像 (BGR)
        返回:
            fg_resized: 缩放后的前景图像
        """
        bh, bw = bg.shape[:2]
        fh, fw = fg.shape[:2]
        # 前景最大不超过背景的60%，保持自然比例
        max_h = int(bh * 0.6)
        max_w = int(bw * 0.6)
        scale = min(max_h / fh, max_w / fw, 1.0)  # 不放大，只缩小
        new_w = max(1, int(fw * scale))
        new_h = max(1, int(fh * scale))
        return cv2.resize(fg, (new_w, new_h), interpolation=cv2.INTER_AREA)

    @staticmethod
    def denoise(img, method='gaussian'):
        """
        对图像进行去噪处理。

        参数:
            img: 输入图像
            method: 'gaussian' 高斯滤波 | 'median' 中值滤波 | 'bilateral' 双边滤波
        返回:
            去噪后的图像
        """
        if method == 'gaussian':
            # 高斯滤波：对高斯噪声效果好，速度快
            return cv2.GaussianBlur(img, (3, 3), 0)
        elif method == 'median':
            # 中值滤波：对椒盐噪声效果好，能保留边缘
            return cv2.medianBlur(img, 3)
        elif method == 'bilateral':
            # 双边滤波：在平滑噪声的同时保留边缘，效果最好但速度最慢
            # d=9: 滤波直径; 75: 颜色空间和空间域的sigma值
            return cv2.bilateralFilter(img, d=9, sigmaColor=75, sigmaSpace=75)
        return img.copy()

    @staticmethod
    def equalize_hist(img):
        """
        对图像进行CLAHE对比度增强（在亮度通道单独处理，避免色偏）。

        参数:
            img: BGR格式输入图像
        返回:
            对比度增强后的图像
        """
        # 转换到LAB颜色空间，只对L通道（亮度）做均衡化
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        # CLAHE: clipLimit=2.0防止过增强, tileGridSize=(8,8)局部均衡化
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l = clahe.apply(l)
        return cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)

    @staticmethod
    def get_steps(fg, bg):
        """
        返回预处理各步骤的中间结果，用于界面中间过程展示。

        返回:
            steps (dict): {步骤名称: 图像}
        """
        steps = {}
        steps['前景原图'] = fg.copy()
        denoised = Preprocessor.denoise(fg, 'bilateral')
        steps['前景去噪(双边)'] = denoised
        enhanced = Preprocessor.equalize_hist(denoised)
        steps['前景增强(CLAHE)'] = enhanced

        steps['背景原图'] = bg.copy()
        bg_denoised = Preprocessor.denoise(bg, 'gaussian')
        steps['背景去噪(高斯)'] = bg_denoised
        return steps


# ============================================================
# 二、前景提取（图像分割）模块
# ============================================================

class ForegroundExtractor:
    """
    从前景图像中分割出目标物体，生成前景掩码（mask）。
    提供三种经典传统方法：
      1. GrabCut：迭代图割算法，效果最好，适合复杂前景
      2. 颜色阈值：在HSV空间按颜色范围分割，适合纯色背景（如绿幕）
      3. Canny + 形态学：边缘检测 + 闭运算填充，适合高对比度边缘场景
    """

    @staticmethod
    def grabcut(img, iterations=5):
        """
        使用GrabCut算法提取前景。
        GrabCut基于图割（Graph Cut）和高斯混合模型（GMM），通过迭代
        优化将像素分类为前景/背景，是传统方法中效果最优的前景提取算法。

        参数:
            img: BGR输入图像
            iterations: GrabCut迭代次数，越多效果越好但越慢
        返回:
            mask_binary: 二值掩码，前景为255，背景为0
            mask_grabcut: GrabCut原始掩码（含可能前景/背景）
        """
        h, w = img.shape[:2]
        # 初始化掩码（全部标记为"可能背景"）
        mask = np.zeros((h, w), np.uint8)
        # GrabCut内部需要的两个临时数组
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        # 初始矩形：留出10%边距，假设目标在图像中心区域
        margin = int(min(h, w) * 0.1)
        rect = (margin, margin, w - 2 * margin, h - 2 * margin)

        # 执行GrabCut：从矩形初始化（INIT_WITH_RECT模式）
        cv2.grabCut(img, mask, rect, bgd_model, fgd_model,
                    iterations, cv2.GC_INIT_WITH_RECT)

        # 将"确定前景"和"可能前景"都视为前景
        # GC_FGD=1（确定前景），GC_PR_FGD=3（可能前景）
        mask_binary = np.where((mask == cv2.GC_FGD) | (mask == cv2.GC_PR_FGD),
                               255, 0).astype(np.uint8)

        # 形态学后处理：去除小噪声，填充孔洞
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_CLOSE, kernel, iterations=3)
        mask_binary = cv2.morphologyEx(mask_binary, cv2.MORPH_OPEN,  kernel, iterations=1)

        return mask_binary, mask

    @staticmethod
    def color_threshold(img, bg_color='green', tolerance=40):
        """
        基于颜色阈值在HSV颜色空间分割前景。
        适用于绿幕（绿色背景）或其他纯色背景场景。

        参数:
            img: BGR输入图像
            bg_color: 背景颜色类型 'green'|'white'|'red'|'blue'|'auto'
            tolerance: 阈值容忍度（越大分割范围越广）
        返回:
            fg_mask: 前景掩码（255=前景，0=背景）
        """
        # 转为HSV颜色空间（更适合颜色分割，分离亮度与色相）
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        h_ch, s_ch, v_ch = cv2.split(hsv)

        if bg_color == 'auto':
            # 自动检测背景颜色：取四角像素颜色取中值估计背景色
            corners = [img[0, 0], img[0, -1], img[-1, 0], img[-1, -1]]
            bg_bgr = np.median(corners, axis=0).astype(np.uint8)
            bg_hsv = cv2.cvtColor(bg_bgr.reshape(1, 1, 3), cv2.COLOR_BGR2HSV)[0, 0]
            h_val = int(bg_hsv[0])
            lo = np.array([max(0, h_val - tolerance), 30, 30])
            hi = np.array([min(179, h_val + tolerance), 255, 255])
            bg_mask = cv2.inRange(hsv, lo, hi)

        elif bg_color == 'green':
            # 绿色背景：HSV色相范围 35-85（绿色区间）
            lo = np.array([35, 40, 40])
            hi = np.array([85, 255, 255])
            bg_mask = cv2.inRange(hsv, lo, hi)

        elif bg_color == 'white':
            # 白色背景：低饱和度 + 高亮度
            lo = np.array([0, 0, 200])
            hi = np.array([179, 40, 255])
            bg_mask = cv2.inRange(hsv, lo, hi)

        elif bg_color == 'red':
            # 红色：HSV中红色跨越0度，需合并两段范围
            lo1 = np.array([0, 80, 80]);   hi1 = np.array([10, 255, 255])
            lo2 = np.array([160, 80, 80]); hi2 = np.array([179, 255, 255])
            bg_mask = cv2.inRange(hsv, lo1, hi1) | cv2.inRange(hsv, lo2, hi2)

        elif bg_color == 'blue':
            lo = np.array([100, 80, 80])
            hi = np.array([130, 255, 255])
            bg_mask = cv2.inRange(hsv, lo, hi)

        else:
            bg_mask = np.zeros(img.shape[:2], np.uint8)

        # 取反得到前景掩码
        fg_mask = cv2.bitwise_not(bg_mask)

        # 形态学后处理：闭运算填充前景内部的小孔，开运算去除背景残余小噪点
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN,  kernel, iterations=1)

        return fg_mask

    @staticmethod
    def canny_morphology(img):
        """
        使用Canny边缘检测 + 形态学操作提取前景。
        适用于前景与背景有较强亮度/颜色对比的场景。

        流程:
            灰度化 → 高斯平滑 → Canny边缘检测 → 膨胀（连接边缘）
            → 轮廓查找 → 选最大轮廓 → 填充得到掩码
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 高斯平滑减少噪声对边缘检测的干扰
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        # Canny双阈值边缘检测：50=低阈值（弱边缘），150=高阈值（强边缘）
        edges = cv2.Canny(blurred, threshold1=50, threshold2=150)

        # 膨胀操作：扩大边缘区域，帮助将断裂的边缘连接成闭合轮廓
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        dilated = cv2.dilate(edges, kernel, iterations=2)

        # 查找所有轮廓，选面积最大的作为主体目标
        contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL,
                                        cv2.CHAIN_APPROX_SIMPLE)
        mask = np.zeros(img.shape[:2], np.uint8)
        if contours:
            # 按轮廓面积排序，取最大的若干轮廓填充
            contours = sorted(contours, key=cv2.contourArea, reverse=True)
            for cnt in contours[:3]:  # 取前3大轮廓，防止只取到噪声
                if cv2.contourArea(cnt) > 500:  # 过滤面积过小的轮廓
                    cv2.drawContours(mask, [cnt], -1, 255, -1)  # -1=填充

        # 闭运算：填充轮廓内部的孔洞
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=3)
        return mask

    @staticmethod
    def refine_mask(mask):
        """
        对掩码进行边缘羽化（Feathering），使前景与背景的边界过渡自然，
        避免出现生硬的锯齿边缘。

        原理：对二值掩码做高斯模糊，使边缘像素的透明度从1逐渐过渡到0。
        """
        # 高斯模糊使边缘像素值从255平滑过渡到0（羽化宽度约5px）
        mask_float = mask.astype(np.float32) / 255.0
        blurred = cv2.GaussianBlur(mask_float, (21, 21), sigmaX=5)
        return np.clip(blurred, 0, 1)


# ============================================================
# 三、图像合成模块
# ============================================================

class ImageCompositor:
    """
    将前景图像（含掩码）合成到背景图像上，提供四种合成策略：
      1. 加权平均合成：最简单，直接按掩码做alpha混合
      2. 多频带融合（Laplacian金字塔）：分别融合低频和高频，过渡最自然
      3. 颜色融合：调整前景颜色以匹配背景色调，减少色差
      4. 泊松融合（近似）：使前景的梯度场无缝嵌入背景
    """

    @staticmethod
    def alpha_blend(fg, bg, mask, position=(0, 0)):
        """
        基础Alpha混合合成。
        公式：result = mask * fg + (1 - mask) * bg

        参数:
            fg: 前景图像（BGR）
            bg: 背景图像（BGR）
            mask: 软掩码，float32，范围[0,1]（1=完全前景，0=完全背景）
            position: 前景放置在背景上的左上角坐标 (x, y)
        返回:
            result: 合成结果图像
        """
        result = bg.copy().astype(np.float32)
        x, y = position
        fh, fw = fg.shape[:2]
        bh, bw = bg.shape[:2]

        # 计算前景在背景中实际可放置的区域（防止越界）
        x1, y1 = max(0, x), max(0, y)
        x2 = min(bw, x + fw)
        y2 = min(bh, y + fh)
        fx1 = x1 - x; fy1 = y1 - y
        fx2 = fx1 + (x2 - x1); fy2 = fy1 + (y2 - y1)

        if x2 <= x1 or y2 <= y1:
            return bg.copy()

        fg_roi  = fg[fy1:fy2, fx1:fx2].astype(np.float32)
        mask_roi = mask[fy1:fy2, fx1:fx2]

        # 将单通道掩码扩展为三通道，与BGR图像对齐
        if mask_roi.ndim == 2:
            alpha = mask_roi[:, :, np.newaxis]
        else:
            alpha = mask_roi

        # Alpha混合公式
        result[y1:y2, x1:x2] = (alpha * fg_roi +
                                  (1.0 - alpha) * result[y1:y2, x1:x2])
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def laplacian_pyramid_blend(fg, bg, mask, position=(0, 0), levels=4):
        """
        多频带融合（Laplacian金字塔融合）。
        原理：将前景、背景、掩码分别分解为高斯金字塔和拉普拉斯金字塔，
        在每个分辨率层级分别做线性融合后再重建图像。
        低频层（模糊版本）使用宽过渡带，高频层（细节）使用窄过渡带，
        结果比单纯Alpha混合更自然，是专业图像合成的标准方法之一。

        参数:
            levels: 金字塔层数（越多融合越平滑，建议3-6层）
        """
        result = bg.copy().astype(np.float32)
        x, y = position
        fh, fw = fg.shape[:2]
        bh, bw = bg.shape[:2]
        x1,y1 = max(0,x), max(0,y)
        x2 = min(bw, x+fw); y2 = min(bh, y+fh)
        if x2 <= x1 or y2 <= y1:
            return bg.copy()
        fx1=x1-x; fy1=y1-y
        fx2=fx1+(x2-x1); fy2=fy1+(y2-y1)

        fg_roi = fg[fy1:fy2, fx1:fx2].astype(np.float32)
        bg_roi = result[y1:y2, x1:x2]
        mask_roi = mask[fy1:fy2, fx1:fx2]
        if mask_roi.ndim == 2:
            mask_roi = mask_roi[:, :, np.newaxis]

        # 确保尺寸适合金字塔（至少能被2^levels整除）
        min_dim = min(fg_roi.shape[:2])
        actual_levels = min(levels, int(np.log2(min_dim)))
        if actual_levels < 1:
            return ImageCompositor.alpha_blend(fg, bg, mask, position)

        # 构建高斯金字塔（逐步下采样）
        def gauss_pyramid(img, lvl):
            gp = [img.copy()]
            for _ in range(lvl):
                gp.append(cv2.pyrDown(gp[-1]))
            return gp

        # 构建拉普拉斯金字塔（相邻层的差，保存高频信息）
        def lap_pyramid(img, lvl):
            gp = gauss_pyramid(img, lvl)
            lp = []
            for i in range(lvl):
                # 将下一层上采样回当前层尺寸
                up = cv2.pyrUp(gp[i+1], dstsize=(gp[i].shape[1], gp[i].shape[0]))
                lp.append(gp[i] - up)       # 高频差值
            lp.append(gp[lvl])              # 最底层保留低频
            return lp

        # 对掩码构建高斯金字塔（用于各层的融合权重）
        def mask_pyramid(m, lvl):
            gp = [m.copy()]
            for _ in range(lvl):
                gp.append(cv2.pyrDown(gp[-1]))
            return gp

        lp_fg   = lap_pyramid(fg_roi,  actual_levels)
        lp_bg   = lap_pyramid(bg_roi,  actual_levels)
        mp_mask = mask_pyramid(mask_roi, actual_levels)

        # 在每一层按该层的掩码融合前景和背景的拉普拉斯系数
        lp_blend = []
        for i in range(actual_levels + 1):
            m = mp_mask[i]
            blended = m * lp_fg[i] + (1.0 - m) * lp_bg[i]
            lp_blend.append(blended)

        # 从最底层开始逐层重建（拉普拉斯金字塔逆变换）
        reconstructed = lp_blend[-1]
        for i in range(actual_levels - 1, -1, -1):
            up = cv2.pyrUp(reconstructed,
                           dstsize=(lp_blend[i].shape[1], lp_blend[i].shape[0]))
            reconstructed = up + lp_blend[i]

        result[y1:y2, x1:x2] = np.clip(reconstructed, 0, 255)
        return result.astype(np.uint8)

    @staticmethod
    def color_match_blend(fg, bg, mask, position=(0, 0)):
        """
        颜色融合合成：先将前景的颜色均值/方差调整到与背景合成区域一致，
        再做Alpha混合。有效减少前背景之间的色调不匹配感。

        原理：在LAB颜色空间（感知线性空间）分别对L/A/B三通道做
        均值和标准差的线性变换（颜色迁移，Reinhard方法）。
        """
        result = bg.copy().astype(np.float32)
        x, y = position
        fh, fw = fg.shape[:2]
        bh, bw = bg.shape[:2]
        x1,y1 = max(0,x), max(0,y)
        x2 = min(bw, x+fw); y2 = min(bh, y+fh)
        if x2 <= x1 or y2 <= y1:
            return bg.copy()
        fx1=x1-x; fy1=y1-y
        fx2=fx1+(x2-x1); fy2=fy1+(y2-y1)

        fg_roi   = fg[fy1:fy2, fx1:fx2]
        bg_roi   = bg[y1:y2, x1:x2]
        mask_roi = mask[fy1:fy2, fx1:fx2]

        # 转到LAB颜色空间做颜色统计迁移（Reinhard et al.）
        fg_lab = cv2.cvtColor(fg_roi,  cv2.COLOR_BGR2LAB).astype(np.float32)
        bg_lab = cv2.cvtColor(bg_roi,  cv2.COLOR_BGR2LAB).astype(np.float32)

        # 对每个通道，将前景的均值/标准差迁移到背景区域的统计量
        for c in range(3):
            fg_mean, fg_std = fg_lab[:,:,c].mean(), fg_lab[:,:,c].std() + 1e-6
            bg_mean, bg_std = bg_lab[:,:,c].mean(), bg_lab[:,:,c].std() + 1e-6
            # 线性变换：先去均值归一化，再按目标分布缩放
            fg_lab[:,:,c] = (fg_lab[:,:,c] - fg_mean) * (bg_std / fg_std) + bg_mean

        # 转回BGR
        fg_lab = np.clip(fg_lab, 0, 255).astype(np.uint8)
        fg_matched = cv2.cvtColor(fg_lab, cv2.COLOR_LAB2BGR)

        # 用颜色匹配后的前景做Alpha混合
        if mask_roi.ndim == 2:
            alpha = mask_roi[:,:,np.newaxis]
        else:
            alpha = mask_roi
        result[y1:y2, x1:x2] = (alpha * fg_matched.astype(np.float32) +
                                  (1 - alpha) * result[y1:y2, x1:x2])
        return np.clip(result, 0, 255).astype(np.uint8)

    @staticmethod
    def poisson_blend(fg, bg, mask_binary, position=(0, 0)):
        """
        泊松融合（Poisson Blending）近似实现。
        OpenCV内置 cv2.seamlessClone 实现了完整的泊松方程求解，
        使前景梯度场无缝嵌入背景，是效果最自然的合成方法。

        参数:
            mask_binary: 二值掩码（uint8，255=前景区域）
        """
        result = bg.copy()
        x, y = position
        fh, fw = fg.shape[:2]
        bh, bw = bg.shape[:2]
        x1,y1 = max(0,x), max(0,y)
        x2 = min(bw, x+fw); y2 = min(bh, y+fh)
        if x2 <= x1 or y2 <= y1:
            return bg.copy()
        fx1=x1-x; fy1=y1-y
        fx2=fx1+(x2-x1); fy2=fy1+(y2-y1)

        fg_roi   = fg[fy1:fy2, fx1:fx2]
        mask_roi = mask_binary[fy1:fy2, fx1:fx2]

        try:
            # cv2.seamlessClone: 泊松方程求解实现无缝克隆
            # MIXED_CLONE模式：混合前景和背景的梯度，在纹理区域保留背景纹理
            # NORMAL_CLONE模式：完全用前景梯度替换背景
            center = (x1 + (x2 - x1) // 2, y1 + (y2 - y1) // 2)
            result = cv2.seamlessClone(fg_roi, result, mask_roi,
                                       center, cv2.NORMAL_CLONE)
        except cv2.error:
            # 如果泊松融合失败（如掩码无效），退化为Alpha混合
            mask_float = ForegroundExtractor.refine_mask(mask_roi)
            result = ImageCompositor.alpha_blend(fg, bg,
                np.pad(mask_float,
                       ((fy1, fh-(fy1+(y2-y1))), (fx1, fw-(fx1+(x2-x1)))),
                       'constant'), position)
        return result


# ============================================================
# 四、后处理增强模块
# ============================================================

class PostProcessor:
    """
    对合成后的图像进行质量增强，弥补合成过程引入的色彩/清晰度问题。
    """

    @staticmethod
    def sharpen(img, strength=1.0):
        """
        非锐化掩蔽（Unsharp Masking）锐化。
        公式: sharpened = img + strength * (img - GaussBlur(img))
        """
        blurred = cv2.GaussianBlur(img, (5, 5), sigmaX=1.0)
        result = cv2.addWeighted(img, 1.0 + strength, blurred, -strength, 0)
        return result

    @staticmethod
    def adjust_contrast_brightness(img, alpha=1.0, beta=0):
        """
        线性对比度和亮度调整。
        公式: dst = alpha * src + beta
        alpha: 对比度倍数（>1增强，<1减弱）
        beta: 亮度偏移（正数变亮，负数变暗）
        """
        return cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    @staticmethod
    def color_adjust(img, hue_shift=0, sat_scale=1.0, val_scale=1.0):
        """
        在HSV空间调整色调、饱和度和明度。
        用于微调合成图像的整体色调，使前景色彩与背景更和谐。
        """
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        hsv[:,:,0] = (hsv[:,:,0] + hue_shift) % 180       # 色调循环
        hsv[:,:,1] = np.clip(hsv[:,:,1] * sat_scale, 0, 255)  # 饱和度
        hsv[:,:,2] = np.clip(hsv[:,:,2] * val_scale, 0, 255)  # 明度
        return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)

    @staticmethod
    def vignette(img, strength=0.4):
        """
        添加暗角效果（Vignette），使视觉焦点集中在图像中心，增强艺术感。
        原理：生成中心亮周边暗的高斯权重图，叠加到图像上。
        """
        h, w = img.shape[:2]
        # 生成两个方向的高斯分布，乘积得到二维暗角掩码
        gx = cv2.getGaussianKernel(w, w * 0.5)
        gy = cv2.getGaussianKernel(h, h * 0.5)
        vignette_mask = gy @ gx.T  # 外积得到二维高斯
        # 归一化到[1-strength, 1]，中心不变，边角变暗
        vignette_mask = vignette_mask / vignette_mask.max()
        vignette_mask = 1.0 - strength * (1.0 - vignette_mask)
        result = img.astype(np.float32)
        for c in range(3):
            result[:,:,c] *= vignette_mask
        return np.clip(result, 0, 255).astype(np.uint8)


# ============================================================
# 五、图形界面（GUI）模块
# ============================================================

class ImageCompositeApp:
    """
    基于Tkinter的图像合成系统主界面。

    界面布局:
    ┌─────────────────────────────────────────────────────────┐
    │ 标题栏                                                   │
    ├──────────────┬──────────────────────────────────────────┤
    │ 控制面板      │ 图像显示区（前景 | 背景 | 合成结果）      │
    │ - 加载图像    │                                          │
    │ - 分割方法    │                                          │
    │ - 合成方法    │                                          │
    │ - 后处理参数  │                                          │
    │ - 位置调节    │                                          │
    │ - 运行/保存   │                                          │
    ├──────────────┴──────────────────────────────────────────┤
    │ 状态栏                                                   │
    └─────────────────────────────────────────────────────────┘
    """

    def __init__(self, root):
        self.root = root
        self.root.title("图像合成系统  |  数字图像处理大作业  |  南阳理工学院")
        self.root.geometry("1280x780")
        self.root.configure(bg='#f0f0f0')

        # 数据
        self.img_fg = None          # 前景图（BGR）
        self.img_bg = None          # 背景图（BGR）
        self.fg_mask = None         # 前景掩码（float32, [0,1]）
        self.fg_mask_binary = None  # 前景掩码（uint8, 0/255）
        self.result = None          # 合成结果

        # 界面变量
        self.seg_method_var    = tk.StringVar(value='grabcut')
        self.seg_bgcol_var     = tk.StringVar(value='auto')
        self.comp_method_var   = tk.StringVar(value='laplacian')
        self.pos_x_var         = tk.IntVar(value=50)
        self.pos_y_var         = tk.IntVar(value=50)
        self.sharp_var         = tk.DoubleVar(value=0.5)
        self.contrast_var      = tk.DoubleVar(value=1.1)
        self.brightness_var    = tk.IntVar(value=0)
        self.sat_var           = tk.DoubleVar(value=1.0)
        self.vignette_var      = tk.DoubleVar(value=0.0)

        self._build_ui()

    # ─── 界面构建 ────────────────────────────────────────────

    def _build_ui(self):
        # 标题栏
        title_fr = tk.Frame(self.root, bg='#1a3c5e', height=48)
        title_fr.pack(fill=tk.X); title_fr.pack_propagate(False)
        tk.Label(title_fr, text="🖼  图像合成系统（传统算法）",
                 font=('Microsoft YaHei', 13, 'bold'),
                 fg='white', bg='#1a3c5e').pack(side=tk.LEFT, padx=20, pady=10)

        # 主体
        main = tk.Frame(self.root, bg='#f0f0f0')
        main.pack(fill=tk.BOTH, expand=True)

        # 左侧控制面板
        ctrl = tk.Frame(main, bg='#e8e8e8', width=260)
        ctrl.pack(side=tk.LEFT, fill=tk.Y, padx=4, pady=4)
        ctrl.pack_propagate(False)
        self._build_control(ctrl)

        # 右侧图像区
        disp = tk.Frame(main, bg='#f0f0f0')
        disp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)
        self._build_display(disp)

        # 状态栏
        self.status = tk.StringVar(value="就绪。请加载前景图像和背景图像。")
        tk.Label(self.root, textvariable=self.status,
                 bd=1, relief=tk.SUNKEN, anchor=tk.W,
                 font=('Microsoft YaHei', 9), fg='#333', bg='#dcdcdc').pack(
            side=tk.BOTTOM, fill=tk.X)

    def _build_control(self, parent):
        def section(text): return ttk.LabelFrame(parent, text=text, padding=6)

        # 文件操作
        f1 = section("📁 加载图像"); f1.pack(fill=tk.X, padx=6, pady=4)
        self._btn(f1, "打开前景图像", self._open_fg, '#1a3c5e')
        self._btn(f1, "打开背景图像", self._open_bg, '#2e6b3e')

        # 前景提取
        f2 = section("✂️ 前景提取方法"); f2.pack(fill=tk.X, padx=6, pady=4)
        for lbl, val in [("GrabCut（推荐）", "grabcut"),
                          ("颜色阈值分割",   "color"),
                          ("Canny+形态学",   "canny")]:
            tk.Radiobutton(f2, text=lbl, variable=self.seg_method_var, value=val,
                           bg='#e8e8e8', font=('Microsoft YaHei', 9)).pack(anchor=tk.W)
        tk.Label(f2, text="背景颜色（颜色阈值用）:", bg='#e8e8e8',
                 font=('Microsoft YaHei', 8)).pack(anchor=tk.W, pady=(4,0))
        bg_color_cb = ttk.Combobox(f2, textvariable=self.seg_bgcol_var,
                                   values=['auto','green','white','red','blue'],
                                   width=12, state='readonly')
        bg_color_cb.pack(anchor=tk.W)
        self._btn(f2, "▶ 提取前景", self._extract_fg, '#7b3a00')

        # 合成方法
        f3 = section("🔀 合成方法"); f3.pack(fill=tk.X, padx=6, pady=4)
        for lbl, val in [("加权Alpha混合",    "alpha"),
                          ("拉普拉斯金字塔",   "laplacian"),
                          ("颜色融合",         "color"),
                          ("泊松融合",         "poisson")]:
            tk.Radiobutton(f3, text=lbl, variable=self.comp_method_var, value=val,
                           bg='#e8e8e8', font=('Microsoft YaHei', 9)).pack(anchor=tk.W)

        # 位置调节
        f4 = section("📐 合成位置"); f4.pack(fill=tk.X, padx=6, pady=4)
        for lbl, var in [("X偏移:", self.pos_x_var), ("Y偏移:", self.pos_y_var)]:
            row = tk.Frame(f4, bg='#e8e8e8'); row.pack(fill=tk.X)
            tk.Label(row, text=lbl, bg='#e8e8e8', font=('Microsoft YaHei', 8),
                     width=7).pack(side=tk.LEFT)
            tk.Scale(row, variable=var, from_=0, to=500, orient=tk.HORIZONTAL,
                     bg='#e8e8e8', length=160, font=('Microsoft YaHei', 7)).pack(side=tk.LEFT)

        # 后处理参数
        f5 = section("✨ 后处理增强"); f5.pack(fill=tk.X, padx=6, pady=4)
        for lbl, var, lo, hi, res in [
            ("锐化强度:", self.sharp_var,      0, 2.0, 0.1),
            ("对比度:",   self.contrast_var,   0.5, 2.0, 0.05),
            ("亮度偏移:", self.brightness_var, -50, 50, 1),
            ("饱和度:",   self.sat_var,        0.5, 2.0, 0.05),
            ("暗角强度:", self.vignette_var,   0, 0.8, 0.05),
        ]:
            row = tk.Frame(f5, bg='#e8e8e8'); row.pack(fill=tk.X)
            tk.Label(row, text=lbl, bg='#e8e8e8', font=('Microsoft YaHei', 8),
                     width=8).pack(side=tk.LEFT)
            tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, bg='#e8e8e8', length=150,
                     font=('Microsoft YaHei', 7)).pack(side=tk.LEFT)

        # 执行按钮
        self._btn(parent, "▶▶  开始合成", self._run_composite, '#c62828', pady=10)
        self._btn(parent, "📊  方法对比", self._compare_all,    '#6a1b9a')
        self._btn(parent, "🔬  展示流程", self._show_pipeline,  '#00695c')
        self._btn(parent, "💾  保存结果", self._save_result,    '#1565c0')

    def _btn(self, parent, text, cmd, color, pady=3):
        tk.Button(parent, text=text, command=cmd, bg=color, fg='white',
                  font=('Microsoft YaHei', 9), relief=tk.FLAT,
                  cursor='hand2').pack(fill=tk.X, padx=6, pady=pady)

    def _build_display(self, parent):
        # 三格显示：前景 | 背景 | 结果
        top = tk.Frame(parent, bg='#f0f0f0')
        top.pack(fill=tk.BOTH, expand=True)

        self.panels = {}
        for title, key in [("前景图（原图）", "fg"),
                            ("背景图", "bg"),
                            ("合成结果", "result")]:
            fr = ttk.LabelFrame(top, text=title, padding=4)
            fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3, pady=3)
            lbl = tk.Label(fr, bg='#bbbbbb', text=f"（{title}）\n\n请加载图像",
                           font=('Microsoft YaHei', 9), fg='#555')
            lbl.pack(fill=tk.BOTH, expand=True)
            info = tk.Label(fr, text="", font=('Microsoft YaHei', 8), bg='#f0f0f0')
            info.pack()
            self.panels[key] = (lbl, info)

    # ─── 功能函数 ─────────────────────────────────────────────

    def _open_fg(self):
        p = filedialog.askopenfilename(title="选择前景图像",
            filetypes=[("图像文件","*.jpg *.jpeg *.png *.bmp"),("全部","*.*")])
        if not p: return
        img = cv2.imread(p)
        if img is None: messagebox.showerror("错误", "无法读取图像"); return
        self.img_fg = img
        self._show(img, 'fg', os.path.basename(p))
        self.status.set(f"已加载前景：{os.path.basename(p)}  {img.shape[1]}×{img.shape[0]}")

    def _open_bg(self):
        p = filedialog.askopenfilename(title="选择背景图像",
            filetypes=[("图像文件","*.jpg *.jpeg *.png *.bmp"),("全部","*.*")])
        if not p: return
        img = cv2.imread(p)
        if img is None: messagebox.showerror("错误", "无法读取图像"); return
        self.img_bg = img
        self._show(img, 'bg', os.path.basename(p))
        self.status.set(f"已加载背景：{os.path.basename(p)}  {img.shape[1]}×{img.shape[0]}")

    def _extract_fg(self):
        if self.img_fg is None:
            messagebox.showwarning("提示", "请先加载前景图像！"); return
        self.status.set("正在提取前景...")
        self.root.update()

        fg = Preprocessor.denoise(self.img_fg, 'bilateral')
        method = self.seg_method_var.get()

        if method == 'grabcut':
            mask_bin, _ = ForegroundExtractor.grabcut(fg)
        elif method == 'color':
            mask_bin = ForegroundExtractor.color_threshold(
                fg, self.seg_bgcol_var.get())
        else:
            mask_bin = ForegroundExtractor.canny_morphology(fg)

        self.fg_mask_binary = mask_bin
        self.fg_mask = ForegroundExtractor.refine_mask(mask_bin)

        # 显示掩码叠加预览
        preview = self.img_fg.copy()
        preview[mask_bin == 0] = (preview[mask_bin == 0] * 0.3).astype(np.uint8)
        self._show(preview, 'fg', f"前景掩码预览（{method}）")
        self.status.set(f"前景提取完成（{method}）。非遮挡区域=前景，暗色区域=背景。")

    def _run_composite(self):
        if self.img_fg is None or self.img_bg is None:
            messagebox.showwarning("提示", "请先加载前景和背景图像！"); return
        if self.fg_mask is None:
            self._extract_fg()

        self.status.set("合成中...")
        self.root.update()

        fg = Preprocessor.denoise(self.img_fg, 'bilateral')
        fg = Preprocessor.resize_to_match(fg, self.img_bg)
        # 同步缩放掩码到与前景一致
        mh, mw = self.fg_mask.shape[:2]
        fh, fw = fg.shape[:2]
        if (mh, mw) != (fh, fw):
            mask_b = cv2.resize(self.fg_mask_binary, (fw, fh))
            mask_f = cv2.resize(self.fg_mask,        (fw, fh))
        else:
            mask_b, mask_f = self.fg_mask_binary, self.fg_mask

        pos = (self.pos_x_var.get(), self.pos_y_var.get())
        method = self.comp_method_var.get()

        if method == 'alpha':
            result = ImageCompositor.alpha_blend(fg, self.img_bg, mask_f, pos)
        elif method == 'laplacian':
            result = ImageCompositor.laplacian_pyramid_blend(
                fg, self.img_bg, mask_f, pos)
        elif method == 'color':
            result = ImageCompositor.color_match_blend(
                fg, self.img_bg, mask_f, pos)
        else:  # poisson
            result = ImageCompositor.poisson_blend(
                fg, self.img_bg, mask_b, pos)

        # 后处理
        result = PostProcessor.adjust_contrast_brightness(
            result, self.contrast_var.get(), self.brightness_var.get())
        if self.sharp_var.get() > 0:
            result = PostProcessor.sharpen(result, self.sharp_var.get())
        if self.sat_var.get() != 1.0:
            result = PostProcessor.color_adjust(result, sat_scale=self.sat_var.get())
        if self.vignette_var.get() > 0:
            result = PostProcessor.vignette(result, self.vignette_var.get())

        self.result = result
        self._show(result, 'result', f"合成结果（{method}）")
        self.status.set(f"合成完成！方法：{method} | 位置：{pos} | 点击【保存结果】导出")

    def _compare_all(self):
        if self.img_fg is None or self.img_bg is None:
            messagebox.showwarning("提示", "请先加载前景和背景图像！"); return
        if self.fg_mask is None:
            self._extract_fg()

        fg = Preprocessor.resize_to_match(
            Preprocessor.denoise(self.img_fg, 'bilateral'), self.img_bg)
        fh, fw = fg.shape[:2]
        mask_b = cv2.resize(self.fg_mask_binary, (fw, fh))
        mask_f = cv2.resize(self.fg_mask,        (fw, fh))
        pos = (self.pos_x_var.get(), self.pos_y_var.get())

        win = tk.Toplevel(self.root); win.title("四种合成方法效果对比")
        win.geometry("960x480"); win.configure(bg='#f0f0f0')
        tk.Label(win, text="四种合成方法对比", font=('Microsoft YaHei',11,'bold'),
                 bg='#f0f0f0').pack(pady=6)
        row = tk.Frame(win, bg='#f0f0f0'); row.pack(fill=tk.BOTH, expand=True, padx=8)

        methods = [('Alpha混合', 'alpha'), ('Laplacian金字塔', 'laplacian'),
                   ('颜色融合',  'color'), ('泊松融合',        'poisson')]
        for name, m in methods:
            if m == 'alpha':     res = ImageCompositor.alpha_blend(fg, self.img_bg, mask_f, pos)
            elif m == 'laplacian': res = ImageCompositor.laplacian_pyramid_blend(fg, self.img_bg, mask_f, pos)
            elif m == 'color':   res = ImageCompositor.color_match_blend(fg, self.img_bg, mask_f, pos)
            else:                res = ImageCompositor.poisson_blend(fg, self.img_bg, mask_b, pos)
            fr = ttk.LabelFrame(row, text=name, padding=4)
            fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=3)
            lbl = tk.Label(fr, bg='#cccccc'); lbl.pack(fill=tk.BOTH, expand=True)
            self._show_in(res, lbl, 200)

    def _show_pipeline(self):
        if self.img_fg is None or self.img_bg is None:
            messagebox.showwarning("提示", "请先加载前景和背景！"); return
        if self.fg_mask is None:
            self._extract_fg()

        steps = Preprocessor.get_steps(self.img_fg, self.img_bg)
        win = tk.Toplevel(self.root); win.title("处理流程中间步骤")
        win.geometry("1100x580"); win.configure(bg='#f0f0f0')
        tk.Label(win, text="图像合成处理流程：各阶段中间结果",
                 font=('Microsoft YaHei',11,'bold'), bg='#f0f0f0').pack(pady=6)

        # 预处理行
        r1 = tk.Frame(win, bg='#f0f0f0'); r1.pack(fill=tk.X, padx=8)
        tk.Label(r1, text="【预处理】", font=('Microsoft YaHei',9,'bold'),
                 fg='#1565c0', bg='#f0f0f0').pack(anchor=tk.W)
        rf = tk.Frame(r1, bg='#f0f0f0'); rf.pack(fill=tk.X)
        for title, img in steps.items():
            f = ttk.LabelFrame(rf, text=title, padding=3); f.pack(side=tk.LEFT, padx=3)
            lbl = tk.Label(f, bg='#ccc'); lbl.pack()
            self._show_in(img, lbl, 150)

        # 分割行
        r2 = tk.Frame(win, bg='#f0f0f0'); r2.pack(fill=tk.X, padx=8, pady=(8,0))
        tk.Label(r2, text="【前景提取】", font=('Microsoft YaHei',9,'bold'),
                 fg='#1b5e20', bg='#f0f0f0').pack(anchor=tk.W)
        rf2 = tk.Frame(r2, bg='#f0f0f0'); rf2.pack(fill=tk.X)
        fg_pre = Preprocessor.denoise(self.img_fg, 'bilateral')
        mask_vis = cv2.cvtColor(self.fg_mask_binary, cv2.COLOR_GRAY2BGR)
        fg_on_mask = self.img_fg.copy()
        fg_on_mask[self.fg_mask_binary == 0] = (
            fg_on_mask[self.fg_mask_binary == 0] * 0.25).astype(np.uint8)
        for title, img in [("前景预处理", fg_pre),
                            ("分割掩码",   mask_vis),
                            ("前景提取效果", fg_on_mask)]:
            f = ttk.LabelFrame(rf2, text=title, padding=3); f.pack(side=tk.LEFT, padx=3)
            lbl = tk.Label(f, bg='#ccc'); lbl.pack()
            self._show_in(img, lbl, 150)

        # 合成结果行
        r3 = tk.Frame(win, bg='#f0f0f0'); r3.pack(fill=tk.X, padx=8, pady=(8,0))
        tk.Label(r3, text="【合成与后处理】", font=('Microsoft YaHei',9,'bold'),
                 fg='#b71c1c', bg='#f0f0f0').pack(anchor=tk.W)
        rf3 = tk.Frame(r3, bg='#f0f0f0'); rf3.pack(fill=tk.X)
        items = [("背景图", self.img_bg)]
        if self.result is not None:
            items += [("合成结果", self.result)]
        for title, img in items:
            f = ttk.LabelFrame(rf3, text=title, padding=3); f.pack(side=tk.LEFT, padx=3)
            lbl = tk.Label(f, bg='#ccc'); lbl.pack()
            self._show_in(img, lbl, 200)

    def _save_result(self):
        if self.result is None:
            messagebox.showwarning("提示", "请先运行合成！"); return
        p = filedialog.asksaveasfilename(
            title="保存合成结果", defaultextension=".png",
            filetypes=[("PNG","*.png"),("JPEG","*.jpg"),("全部","*.*")])
        if p:
            cv2.imwrite(p, self.result)
            messagebox.showinfo("成功", f"已保存：{p}")
            self.status.set(f"已保存：{p}")

    # ─── 辅助 ──────────────────────────────────────────────────

    def _show(self, img, key, name=""):
        lbl, info = self.panels[key]
        self._show_in(img, lbl)
        h, w = img.shape[:2]
        info.config(text=f"{w}×{h}  {name}")

    def _show_in(self, img, label, max_size=None):
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if max_size:
            pil.thumbnail((max_size, max_size), Image.LANCZOS)
        else:
            label.update_idletasks()
            w = label.winfo_width() or 380
            h = label.winfo_height() or 300
            pil.thumbnail((w-8, h-8), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        label.config(image=photo, text='')
        label.image = photo


# ============================================================
# 程序入口
# ============================================================

def main():
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        style.theme_use('clam')
    except Exception:
        pass
    app = ImageCompositeApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()


