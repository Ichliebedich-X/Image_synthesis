import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
from PIL import Image, ImageTk
import os
import threading


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
            if m.ndim == 2:
                m = m[:, :, np.newaxis]
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
    """

    def __init__(self, root):
        self.root = root
        self.root.title("图像合成系统  |  数字图像处理大作业  |  南阳理工学院")
        self.root.geometry("1360x820")
        self.root.minsize(1024, 680)
        self.root.configure(bg='#f0f4f8')

        # 数据
        self.img_fg = None          # 前景图（BGR）
        self.img_bg = None          # 背景图（BGR）
        self.fg_mask = None         # 前景掩码（float32, [0,1]）
        self.fg_mask_binary = None  # 前景掩码（uint8, 0/255）
        self.result = None          # 合成结果

        # 缓存：避免重复预处理
        self._cache_denoised_fg = None
        self._cache_denoised_bg = None

        # 线程控制
        self._processing_thread = None
        self._is_processing = False

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
        # ── 顶部标题栏 ──
        title_fr = tk.Frame(self.root, bg='#1a3c5e', height=52)
        title_fr.pack(fill=tk.X); title_fr.pack_propagate(False)
        tk.Label(title_fr, text=" 图像合成系统（传统算法）",
                 font=('Microsoft YaHei', 14, 'bold'),
                 fg='white', bg='#1a3c5e').pack(side=tk.LEFT, padx=22, pady=12)

        # ── 进度条（初始隐藏，放置于控制面板中） ──
        # 实际创建在 _build_control 中
        self.progress = None

        # ── 主体区域 ──
        main = tk.Frame(self.root, bg='#f0f4f8')
        main.pack(fill=tk.BOTH, expand=True)

        # ── 左侧控制面板（含滚动条） ──
        ctrl_canvas = tk.Canvas(main, bg='#e8edf2', width=300, highlightthickness=0)
        ctrl_scroll = ttk.Scrollbar(main, orient=tk.VERTICAL, command=ctrl_canvas.yview)
        ctrl_inner = tk.Frame(ctrl_canvas, bg='#e8edf2')

        ctrl_inner.bind('<Configure>',
            lambda e: ctrl_canvas.configure(scrollregion=ctrl_canvas.bbox('all')))
        ctrl_canvas.create_window((0, 0), window=ctrl_inner, anchor='nw', width=295)
        ctrl_canvas.configure(yscrollcommand=ctrl_scroll.set)

        ctrl_canvas.pack(side=tk.LEFT, fill=tk.Y, padx=(6, 0), pady=6)
        ctrl_scroll.pack(side=tk.LEFT, fill=tk.Y, pady=6)
        self._build_control(ctrl_inner)

        # ── 右侧图像显示区 ──
        disp = tk.Frame(main, bg='#f0f4f8')
        disp.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self._build_display(disp)

        # ── 底部状态栏 ──
        status_bg = '#1e3a5f'
        self.status = tk.StringVar(value="就绪。请加载前景图像和背景图像。")
        status_frame = tk.Frame(self.root, bg=status_bg, height=32)
        status_frame.pack(side=tk.BOTTOM, fill=tk.X); status_frame.pack_propagate(False)
        tk.Label(status_frame, textvariable=self.status,
                 anchor=tk.W, font=('Microsoft YaHei', 9),
                 fg='#cfdfff', bg=status_bg).pack(side=tk.LEFT, padx=14, pady=4)
        # 状态栏右侧提示
        tk.Label(status_frame, text="南阳理工学院 · 数字图像处理大作业",
                 font=('Microsoft YaHei', 8), fg='#6f8fc0', bg=status_bg).pack(
            side=tk.RIGHT, padx=14, pady=4)

    def _build_control(self, parent):
        """构建左侧控制面板——所有控件布局在此。"""
        def section(text):
            return ttk.LabelFrame(parent, text=text, padding=8)

        # ── 1. 加载图像 ──
        f1 = section(" 加载图像"); f1.pack(fill=tk.X, padx=8, pady=(6, 3))
        self._btn(f1, "打开前景图像", self._open_fg, '#1a3c5e')
        self._btn(f1, "打开背景图像", self._open_bg, '#2e6b3e')

        # ── 2. 前景提取 ──
        f2 = section(" 前景提取方法"); f2.pack(fill=tk.X, padx=8, pady=3)
        for lbl, val in [("GrabCut（推荐）", "grabcut"),
                          ("颜色阈值分割",   "color"),
                          ("Canny+形态学",   "canny")]:
            tk.Radiobutton(f2, text=lbl, variable=self.seg_method_var, value=val,
                           bg='#e8edf2', font=('Microsoft YaHei', 9),
                           activebackground='#d0d8e0').pack(anchor=tk.W, pady=1)
        sep = tk.Frame(f2, bg='#c0c8d0', height=1); sep.pack(fill=tk.X, pady=4)
        lbl_bg = tk.Label(f2, text="背景颜色（颜色阈值用）:",
                          bg='#e8edf2', font=('Microsoft YaHei', 8))
        lbl_bg.pack(anchor=tk.W, pady=(0, 2))
        bg_color_cb = ttk.Combobox(f2, textvariable=self.seg_bgcol_var,
                                   values=['auto','green','white','red','blue'],
                                   width=14, state='readonly')
        bg_color_cb.pack(anchor=tk.W, pady=(0, 4))
        self._btn(f2, "提取前景", self._extract_fg, '#7b3a00')

        # ── 3. 合成方法 ──
        f3 = section(" 合成方法"); f3.pack(fill=tk.X, padx=8, pady=3)
        for lbl, val in [("加权 Alpha 混合",    "alpha"),
                          ("拉普拉斯金字塔融合", "laplacian"),
                          ("颜色匹配融合",       "color"),
                          ("泊松融合（无缝）",   "poisson")]:
            tk.Radiobutton(f3, text=lbl, variable=self.comp_method_var, value=val,
                           bg='#e8edf2', font=('Microsoft YaHei', 9),
                           activebackground='#d0d8e0').pack(anchor=tk.W, pady=1)

        # ── 4. 合成位置 ──
        f4 = section(" 合成位置"); f4.pack(fill=tk.X, padx=8, pady=3)
        for lbl, var in [("X 偏移:", self.pos_x_var), ("Y 偏移:", self.pos_y_var)]:
            row = tk.Frame(f4, bg='#e8edf2'); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=lbl, bg='#e8edf2', font=('Microsoft YaHei', 9),
                     width=7, anchor=tk.E).pack(side=tk.LEFT)
            tk.Scale(row, variable=var, from_=0, to=500, orient=tk.HORIZONTAL,
                     bg='#e8edf2', troughcolor='#c0c8d0', length=180,
                     font=('Microsoft YaHei', 7), sliderlength=18).pack(side=tk.LEFT, padx=4)

        # ── 5. 后处理增强 ──
        f5 = section(" 后处理增强"); f5.pack(fill=tk.X, padx=8, pady=3)
        params = [
            ("锐化强度:", self.sharp_var,      0, 2.0, 0.1),
            ("对比度:",   self.contrast_var,   0.5, 2.0, 0.05),
            ("亮度偏移:", self.brightness_var, -50, 50, 1),
            ("饱和度:",   self.sat_var,        0.5, 2.0, 0.05),
            ("暗角强度:", self.vignette_var,   0, 0.8, 0.05),
        ]
        for lbl, var, lo, hi, res in params:
            row = tk.Frame(f5, bg='#e8edf2'); row.pack(fill=tk.X, pady=1)
            tk.Label(row, text=lbl, bg='#e8edf2', font=('Microsoft YaHei', 9),
                     width=9, anchor=tk.E).pack(side=tk.LEFT)
            tk.Scale(row, variable=var, from_=lo, to=hi, resolution=res,
                     orient=tk.HORIZONTAL, bg='#e8edf2', troughcolor='#c0c8d0',
                     length=170, font=('Microsoft YaHei', 7),
                     sliderlength=18).pack(side=tk.LEFT, padx=4)

        # ── 6. 进度条 ──
        self.progress = ttk.Progressbar(parent, mode='indeterminate', length=280)
        # 不 pack，需要时再显示

        # ── 7. 操作按钮区 ──
        sep2 = tk.Frame(parent, bg='#c0c8d0', height=1); sep2.pack(fill=tk.X, padx=8, pady=6)
        btn_frame = tk.Frame(parent, bg='#e8edf2')
        btn_frame.pack(fill=tk.X, padx=8, pady=(0, 8))
        self.progress_btn_frame = btn_frame

        self._btn(btn_frame, "开始合成", self._run_composite, '#c62828', pady=6)
        self._btn(btn_frame, "四种方法对比", self._compare_all, '#6a1b9a', pady=4)
        self._btn(btn_frame, "查看处理流程", self._show_pipeline, '#00695c', pady=4)
        self._btn(btn_frame, "保存结果", self._save_result, '#1565c0', pady=6)

    def _btn(self, parent, text, cmd, color, pady=4):
        """统一创建样式一致的按钮。"""
        btn = tk.Button(parent, text=text, command=cmd,
                        bg=color, fg='white',
                        activebackground=self._lighten_color(color, 0.2),
                        activeforeground='white',
                        font=('Microsoft YaHei', 9, 'bold'),
                        relief=tk.FLAT, bd=0,
                        cursor='hand2', padx=8)
        btn.pack(fill=tk.X, padx=6, pady=pady, ipady=4)
        return btn

    @staticmethod
    def _lighten_color(hex_color, factor=0.2):
        """将十六进制颜色变亮，用于按钮 hover 效果。"""
        hex_color = hex_color.lstrip('#')
        r, g, b = int(hex_color[0:2], 16), int(hex_color[2:4], 16), int(hex_color[4:6], 16)
        r = min(255, int(r + (255 - r) * factor))
        g = min(255, int(g + (255 - g) * factor))
        b = min(255, int(b + (255 - b) * factor))
        return f'#{r:02x}{g:02x}{b:02x}'

    def _build_display(self, parent):
        """构建右侧三格图片显示面板。"""
        top = tk.Frame(parent, bg='#f0f4f8')
        top.pack(fill=tk.BOTH, expand=True)

        self.panels = {}
        colors = {'fg': '#e3f0ff', 'bg': '#e8f5e9', 'result': '#fff3e0'}
        for title, key in [("前景图（原图）", "fg"),
                            ("背景图", "bg"),
                            ("合成结果", "result")]:
            # 外框：加浅色边框模拟卡片效果
            outer = tk.Frame(top, bg='#ffffff', bd=1, relief=tk.SOLID)
            outer.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4, pady=4)

            fr = ttk.LabelFrame(outer, text=title, padding=6)
            fr.pack(fill=tk.BOTH, expand=True, padx=1, pady=1)

            # 图片显示区域
            lbl = tk.Label(fr, bg=colors[key], text=f"（{title}）\n\n请加载图像",
                           font=('Microsoft YaHei', 10), fg='#666')
            lbl.pack(fill=tk.BOTH, expand=True)

            # 底部分隔线 + 信息栏
            sep_line = tk.Frame(fr, bg='#d0d8e0', height=1)
            sep_line.pack(fill=tk.X)
            info_bg = '#f5f5f5'
            info = tk.Label(fr, text="等待加载……", font=('Microsoft YaHei', 8),
                            bg=info_bg, fg='#888', anchor=tk.W)
            info.pack(fill=tk.X, ipady=2)

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

    def _set_controls_enabled(self, enabled):
        """启用/禁用所有操作按钮，防止处理期间重复点击。"""
        for child in self.progress_btn_frame.winfo_children():
            if isinstance(child, tk.Button):
                child.config(state=tk.NORMAL if enabled else tk.DISABLED)

    def _show_progress(self, show=True):
        """显示/隐藏进度条。"""
        if show and self.progress is not None:
            self.progress.pack(fill=tk.X, padx=8, pady=(0, 2))
            self.progress.start(10)
        elif self.progress is not None:
            self.progress.stop()
            self.progress.pack_forget()

    def _extract_fg(self):
        """异步执行前景提取，避免阻塞界面。"""
        if self.img_fg is None:
            messagebox.showwarning("提示", "请先加载前景图像！"); return
        if self._is_processing:
            return

        self._is_processing = True
        self._set_controls_enabled(False)
        self._show_progress(True)
        self.status.set("正在提取前景...")

        def task():
            try:
                fg = Preprocessor.denoise(self.img_fg, 'bilateral')
                self._cache_denoised_fg = fg
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

                preview = self.img_fg.copy()
                preview[mask_bin == 0] = (preview[mask_bin == 0] * 0.3).astype(np.uint8)

                def done():
                    self._show(preview, 'fg', f"前景掩码预览（{method}）")
                    self.status.set(f"前景提取完成（{method}）。非遮挡区域=前景，暗色区域=背景。")
                    self._show_progress(False)
                    self._set_controls_enabled(True)
                    self._is_processing = False
                self.root.after(0, done)
            except Exception as e:
                def err():
                    messagebox.showerror("提取出错", str(e))
                    self.status.set("前景提取失败")
                    self._show_progress(False)
                    self._set_controls_enabled(True)
                    self._is_processing = False
                self.root.after(0, err)

        threading.Thread(target=task, daemon=True).start()

    def _run_composite(self):
        """异步执行图像合成，避免阻塞界面。"""
        if self.img_fg is None or self.img_bg is None:
            messagebox.showwarning("提示", "请先加载前景和背景图像！"); return
        if self._is_processing:
            return
        if self.fg_mask is None:
            messagebox.showwarning("提示", "请先点击「提取前景」提取前景掩码！")
            return

        self._is_processing = True
        self._set_controls_enabled(False)
        self._show_progress(True)
        self.status.set("正在合成中，请稍候...")

        def task():
            try:
                # 使用缓存的去噪结果，避免重复计算
                fg = self._cache_denoised_fg if self._cache_denoised_fg is not None else \
                     Preprocessor.denoise(self.img_fg, 'bilateral')
                fg_resized = Preprocessor.resize_to_match(fg, self.img_bg)

                # 同步缩放掩码到与前景一致
                mh, mw = self.fg_mask.shape[:2]
                fh, fw = fg_resized.shape[:2]
                if (mh, mw) != (fh, fw):
                    mask_b = cv2.resize(self.fg_mask_binary, (fw, fh))
                    mask_f = cv2.resize(self.fg_mask,        (fw, fh))
                else:
                    mask_b, mask_f = self.fg_mask_binary, self.fg_mask

                pos = (self.pos_x_var.get(), self.pos_y_var.get())
                method = self.comp_method_var.get()

                if method == 'alpha':
                    result = ImageCompositor.alpha_blend(fg_resized, self.img_bg, mask_f, pos)
                elif method == 'laplacian':
                    result = ImageCompositor.laplacian_pyramid_blend(
                        fg_resized, self.img_bg, mask_f, pos)
                elif method == 'color':
                    result = ImageCompositor.color_match_blend(
                        fg_resized, self.img_bg, mask_f, pos)
                else:  # poisson
                    result = ImageCompositor.poisson_blend(
                        fg_resized, self.img_bg, mask_b, pos)

                # 后处理
                result = PostProcessor.adjust_contrast_brightness(
                    result, self.contrast_var.get(), self.brightness_var.get())
                if self.sharp_var.get() > 0:
                    result = PostProcessor.sharpen(result, self.sharp_var.get())
                if self.sat_var.get() != 1.0:
                    result = PostProcessor.color_adjust(result, sat_scale=self.sat_var.get())
                if self.vignette_var.get() > 0:
                    result = PostProcessor.vignette(result, self.vignette_var.get())

                def done():
                    self.result = result
                    self._show(result, 'result', f"合成结果（{method}）")
                    self.status.set(f"合成完成！方法：{method} | 位置：{pos}")
                    self._show_progress(False)
                    self._set_controls_enabled(True)
                    self._is_processing = False
                self.root.after(0, done)
            except Exception as e:
                def err():
                    messagebox.showerror("合成出错", str(e))
                    self.status.set("合成失败")
                    self._show_progress(False)
                    self._set_controls_enabled(True)
                    self._is_processing = False
                self.root.after(0, err)

        threading.Thread(target=task, daemon=True).start()

    def _compare_all(self):
        """异步对比四种合成方法，在弹出窗口中展示。"""
        if self.img_fg is None or self.img_bg is None:
            messagebox.showwarning("提示", "请先加载前景和背景图像！"); return
        if self._is_processing:
            return
        if self.fg_mask is None:
            messagebox.showwarning("提示", "请先点击「提取前景」提取前景掩码！")
            return

        self._is_processing = True
        self._set_controls_enabled(False)
        self._show_progress(True)
        self.status.set("正在对比四种合成方法...")

        def task():
            try:
                fg = self._cache_denoised_fg if self._cache_denoised_fg is not None else \
                     Preprocessor.denoise(self.img_fg, 'bilateral')
                fg = Preprocessor.resize_to_match(fg, self.img_bg)
                fh, fw = fg.shape[:2]
                mask_b = cv2.resize(self.fg_mask_binary, (fw, fh))
                mask_f = cv2.resize(self.fg_mask,        (fw, fh))
                pos = (self.pos_x_var.get(), self.pos_y_var.get())

                methods = [('Alpha 混合', 'alpha'),
                           ('Laplacian 金字塔', 'laplacian'),
                           ('颜色匹配融合', 'color'),
                           ('泊松融合（无缝）', 'poisson')]
                results_data = []
                for name, m in methods:
                    if m == 'alpha':
                        res = ImageCompositor.alpha_blend(fg, self.img_bg, mask_f, pos)
                    elif m == 'laplacian':
                        res = ImageCompositor.laplacian_pyramid_blend(fg, self.img_bg, mask_f, pos)
                    elif m == 'color':
                        res = ImageCompositor.color_match_blend(fg, self.img_bg, mask_f, pos)
                    else:
                        res = ImageCompositor.poisson_blend(fg, self.img_bg, mask_b, pos)
                    results_data.append((name, res))

                def done():
                    win = tk.Toplevel(self.root)
                    win.title("四种合成方法效果对比")
                    win.geometry("1100x520")
                    win.configure(bg='#f0f4f8')
                    win.minsize(800, 400)

                    header_bg = tk.Frame(win, bg='#1a3c5e', height=40)
                    header_bg.pack(fill=tk.X); header_bg.pack_propagate(False)
                    tk.Label(header_bg, text="四种合成方法效果对比",
                             font=('Microsoft YaHei', 12, 'bold'),
                             fg='white', bg='#1a3c5e').pack(pady=8)

                    row = tk.Frame(win, bg='#f0f4f8')
                    row.pack(fill=tk.BOTH, expand=True, padx=8, pady=8)

                    for name, res in results_data:
                        fr = ttk.LabelFrame(row, text=name, padding=6)
                        fr.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=4)
                        lbl = tk.Label(fr, bg='#e8edf2')
                        lbl.pack(fill=tk.BOTH, expand=True)
                        self._show_in(res, lbl, 240)

                    self.status.set("方法对比完成")
                    self._show_progress(False)
                    self._set_controls_enabled(True)
                    self._is_processing = False
                self.root.after(0, done)
            except Exception as e:
                def err():
                    messagebox.showerror("对比出错", str(e))
                    self._show_progress(False)
                    self._set_controls_enabled(True)
                    self._is_processing = False
                self.root.after(0, err)

        threading.Thread(target=task, daemon=True).start()

    def _show_pipeline(self):
        """展示处理流程各阶段中间结果。"""
        if self.img_fg is None or self.img_bg is None:
            messagebox.showwarning("提示", "请先加载前景和背景！"); return
        if self.fg_mask is None:
            messagebox.showwarning("提示", "请先点击「提取前景」提取前景掩码！")
            return

        steps = Preprocessor.get_steps(self.img_fg, self.img_bg)
        win = tk.Toplevel(self.root)
        win.title("处理流程中间步骤")
        bg_color = '#f0f4f8'
        win.geometry("1160x640")
        win.configure(bg=bg_color)
        win.minsize(900, 500)

        # 标题栏
        header = tk.Frame(win, bg='#1a3c5e', height=38)
        header.pack(fill=tk.X); header.pack_propagate(False)
        tk.Label(header, text="图像合成处理流程 — 各阶段中间结果",
                 font=('Microsoft YaHei', 11, 'bold'),
                 fg='white', bg='#1a3c5e').pack(pady=7)

        body = tk.Frame(win, bg=bg_color)
        body.pack(fill=tk.BOTH, expand=True, padx=10, pady=8)

        # ── 预处理行 ──
        r1 = tk.Frame(body, bg=bg_color); r1.pack(fill=tk.X, pady=(0, 6))
        tk.Label(r1, text="【预处理】", font=('Microsoft YaHei', 10, 'bold'),
                 fg='#1565c0', bg=bg_color).pack(anchor=tk.W, pady=(0, 2))
        rf = tk.Frame(r1, bg=bg_color); rf.pack(fill=tk.X)
        for title, img in steps.items():
            f = ttk.LabelFrame(rf, text=title, padding=4)
            f.pack(side=tk.LEFT, padx=3, ipady=2)
            lbl = tk.Label(f, bg='#e3f0ff')
            lbl.pack()
            self._show_in(img, lbl, 160)

        # ── 分割行 ──
        r2 = tk.Frame(body, bg=bg_color); r2.pack(fill=tk.X, pady=6)
        tk.Label(r2, text="【前景提取】", font=('Microsoft YaHei', 10, 'bold'),
                 fg='#1b5e20', bg=bg_color).pack(anchor=tk.W, pady=(0, 2))
        rf2 = tk.Frame(r2, bg=bg_color); rf2.pack(fill=tk.X)
        fg_pre = Preprocessor.denoise(self.img_fg, 'bilateral')
        mask_vis = cv2.cvtColor(self.fg_mask_binary, cv2.COLOR_GRAY2BGR)
        fg_on_mask = self.img_fg.copy()
        fg_on_mask[self.fg_mask_binary == 0] = (
            fg_on_mask[self.fg_mask_binary == 0] * 0.25).astype(np.uint8)
        for title, img in [("前景预处理", fg_pre),
                            ("分割掩码",   mask_vis),
                            ("前景提取效果", fg_on_mask)]:
            f = ttk.LabelFrame(rf2, text=title, padding=4)
            f.pack(side=tk.LEFT, padx=3)
            lbl = tk.Label(f, bg='#e8f5e9')
            lbl.pack()
            self._show_in(img, lbl, 160)

        # ── 合成结果行 ──
        r3 = tk.Frame(body, bg=bg_color); r3.pack(fill=tk.X, pady=6)
        tk.Label(r3, text="【合成与后处理】", font=('Microsoft YaHei', 10, 'bold'),
                 fg='#b71c1c', bg=bg_color).pack(anchor=tk.W, pady=(0, 2))
        rf3 = tk.Frame(r3, bg=bg_color); rf3.pack(fill=tk.X)
        items = [("背景图", self.img_bg)]
        if self.result is not None:
            items += [("合成结果", self.result)]
        for title, img in items:
            f = ttk.LabelFrame(rf3, text=title, padding=4)
            f.pack(side=tk.LEFT, padx=3)
            lbl = tk.Label(f, bg='#fff3e0')
            lbl.pack()
            self._show_in(img, lbl, 220)

    def _save_result(self):
        """增强版保存：先预览确认，再保存文件。"""
        if self.result is None:
            messagebox.showwarning("提示", "请先运行合成！"); return

        # 计算文件信息
        h, w = self.result.shape[:2]
        est_size = w * h * 3 / 1024  # 近似 KB（BGR 3通道）

        # 弹出保存前预览对话框
        preview_win = tk.Toplevel(self.root)
        preview_win.title("保存合成结果 - 预览确认")
        preview_win.geometry("480x420")
        preview_win.configure(bg='#f0f4f8')
        preview_win.resizable(False, False)
        preview_win.transient(self.root)
        preview_win.grab_set()

        tk.Label(preview_win, text="保存前预览",
                 font=('Microsoft YaHei', 12, 'bold'),
                 bg='#f0f4f8', fg='#1a3c5e').pack(pady=(10, 4))

        # 缩略图
        preview_img = cv2.cvtColor(self.result, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(preview_img)
        pil_img.thumbnail((320, 200), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil_img)
        img_lbl = tk.Label(preview_win, image=photo, bg='#ffffff',
                           bd=1, relief=tk.SOLID)
        img_lbl.image = photo
        img_lbl.pack(pady=6)

        # 信息面板
        info_fr = tk.Frame(preview_win, bg='#e8edf2', bd=1, relief=tk.SOLID)
        info_fr.pack(fill=tk.X, padx=30, pady=6)
        info_text = (
            f"尺寸：{w} × {h} 像素\n"
            f"通道：BGR（3通道）\n"
            f"估计文件大小：{est_size:.0f} KB（PNG） / {est_size//3:.0f} KB（JPEG）\n"
            f"当前合成方法：{self.comp_method_var.get()}"
        )
        tk.Label(info_fr, text=info_text, bg='#e8edf2',
                 font=('Microsoft YaHei', 9), fg='#333',
                 justify=tk.LEFT, anchor=tk.W).pack(padx=12, pady=8, fill=tk.X)

        # 按钮区
        btn_fr = tk.Frame(preview_win, bg='#f0f4f8')
        btn_fr.pack(fill=tk.X, padx=30, pady=(6, 12))

        user_choice = {'path': None}

        def on_confirm():
            preview_win.destroy()
            p = filedialog.asksaveasfilename(
                title="保存合成结果", defaultextension=".png",
                filetypes=[("PNG 图片", "*.png"),
                           ("JPEG 图片", "*.jpg"),
                           ("BMP 图片", "*.bmp"),
                           ("全部文件", "*.*")])
            if p:
                user_choice['path'] = p

        def on_cancel():
            preview_win.destroy()

        tk.Button(btn_fr, text="选择路径并保存", command=on_confirm,
                  bg='#1565c0', fg='white', font=('Microsoft YaHei', 9, 'bold'),
                  activebackground='#1976d2', activeforeground='white',
                  relief=tk.FLAT, padx=12, cursor='hand2').pack(side=tk.LEFT, padx=4)
        tk.Button(btn_fr, text="取消", command=on_cancel,
                  bg='#888888', fg='white', font=('Microsoft YaHei', 9),
                  activebackground='#999999', activeforeground='white',
                  relief=tk.FLAT, padx=12, cursor='hand2').pack(side=tk.RIGHT, padx=4)

        self.root.wait_window(preview_win)

        p = user_choice['path']
        if not p:
            return

        success = cv2.imwrite(p, self.result)
        if success:
            file_size = os.path.getsize(p) / 1024
            messagebox.showinfo("保存成功",
                f"文件已保存至：\n{p}\n\n"
                f"文件大小：{file_size:.1f} KB\n"
                f"图像尺寸：{w} × {h}")
            self.status.set(f"已保存：{os.path.basename(p)} ({file_size:.0f} KB)")
        else:
            messagebox.showerror("保存失败", "无法写入文件，请检查路径权限。")

    # ─── 辅助 ──────────────────────────────────────────────────

    def _show(self, img, key, name=""):
        """在主面板中显示图像并更新底部信息。"""
        lbl, info = self.panels[key]
        self._show_in(img, lbl)
        h, w = img.shape[:2]
        info.config(text=f"{w} × {h}  {name}")

    def _show_in(self, img, label, max_size=None):
        """将 OpenCV BGR 图像缩放到 Label 中显示，保持宽高比。"""
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        pil = Image.fromarray(rgb)
        if max_size:
            pil.thumbnail((max_size, max_size), Image.LANCZOS)
        else:
            label.update_idletasks()
            w = label.winfo_width() or 380
            h = label.winfo_height() or 280
            pil.thumbnail((w - 10, h - 10), Image.LANCZOS)
        photo = ImageTk.PhotoImage(pil)
        label.config(image=photo, text='', bg='#ffffff')
        label.image = photo


# ============================================================
# 程序入口
# ============================================================

def main():
    root = tk.Tk()
    try:
        style = ttk.Style(root)
        style.theme_use('clam')
        # 自定义 LabelFrame 样式
        style.configure('TLabelframe', background='#e8edf2', relief=tk.FLAT, bd=0)
        style.configure('TLabelframe.Label', background='#e8edf2',
                        font=('Microsoft YaHei', 9, 'bold'), foreground='#1a3c5e')
        # 自定义 Combobox 样式
        style.configure('TCombobox', font=('Microsoft YaHei', 9), fieldbackground='white')
        # 自定义 Progressbar 样式
        style.configure('Horizontal.TProgressbar', background='#1565c0',
                        troughcolor='#d0d8e0', bordercolor='#e8edf2',
                        lightcolor='#42a5f5', darkcolor='#1565c0')
        # 自定义 Scrollbar 样式
        style.configure('Vertical.TScrollbar', background='#c0c8d0',
                        troughcolor='#e8edf2', bordercolor='#e8edf2',
                        arrowcolor='#555555')
    except Exception:
        pass
    app = ImageCompositeApp(root)
    root.mainloop()


if __name__ == '__main__':
    main()


