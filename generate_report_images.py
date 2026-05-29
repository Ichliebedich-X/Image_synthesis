"""
为实验报告生成所有步骤图（process_前缀，每步一图）。
Run: python generate_report_images.py
输出到 D:/report_images/ 目录。
"""
import sys
sys.path.insert(0, 'D:\\AiStdio\\Python\\PythonProject\\DIP')

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont
import os
from Image_synthesis import (
    Preprocessor, ForegroundExtractor, ImageCompositor, PostProcessor
)

OUT = r'D:\report_images'
os.makedirs(OUT, exist_ok=True)

FG_PATH = r'D:\AiStdio\Python\PythonProject\DIP\dataset\foreground\person_greenscreen.jpg'
BG_PATH = r'D:\AiStdio\Python\PythonProject\DIP\dataset\background\outdoor_sky.jpg'
BG_INDOOR = r'D:\AiStdio\Python\PythonProject\DIP\dataset\background\indoor_room.jpg'
BG_NIGHT = r'D:\AiStdio\Python\PythonProject\DIP\dataset\background\night_gradient.jpg'
FG2_PATH = r'D:\AiStdio\Python\PythonProject\DIP\dataset\foreground\flower_whitebg.jpg'

SEQ = 0

def save(img, name):
    """保存图像到输出目录"""
    global SEQ
    SEQ += 1
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print(f'  [{SEQ:02d}] Saved: {name} ({img.shape[1]}x{img.shape[0]})')
    return path

def put_text(img, text, pos=(10, 25), scale=0.7, color=(255,255,255), thickness=2, bg=None):
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil_img)
    font_size = max(10, int(scale * 28))
    font = _get_cn_font(font_size)
    if bg is not None:
        bbox = draw.textbbox(pos, text, font=font)
        draw.rectangle([bbox[0]-2, bbox[1]-2, bbox[2]+2, bbox[3]+2], fill=tuple(reversed(bg)))
    r, g, b = color
    draw.text(pos, text, fill=(b, g, r), font=font)
    img[:] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)

_FONT_CACHE = {}
def _get_cn_font(size):
    if size not in _FONT_CACHE:
        for p in [r'C:\Windows\Fonts\simsun.ttc', r'C:\Windows\Fonts\msyh.ttc']:
            if os.path.exists(p):
                _FONT_CACHE[size] = ImageFont.truetype(p, size)
                break
        else:
            _FONT_CACHE[size] = ImageFont.load_default()
    return _FONT_CACHE[size]

def hstack_resize(imgs, target_h=300):
    imgs_r = []
    for img in imgs:
        h, w = img.shape[:2]
        ratio = target_h / h
        imgs_r.append(cv2.resize(img, (int(w*ratio), target_h)))
    return np.hstack(imgs_r)

def make_compare_img(imgs, labels, target_h=250, font_scale=0.6):
    panels = []
    for img, label in zip(imgs, labels):
        h, w = img.shape[:2]
        r = target_h / h
        panel = cv2.resize(img, (int(w*r), target_h))
        put_text(panel, label, (8, 22), font_scale, (255,255,255), 2, (0,0,0))
        panels.append(panel)
    return np.hstack(panels)

def gray_to_bgr(img_gray):
    return cv2.cvtColor(img_gray, cv2.COLOR_GRAY2BGR)


print('=' * 60)
print('生成实验报告所需步骤图...')
print('=' * 60)

# ── 1. 加载测试图像 ──
print('\n[3.2.1] 预处理流程')
fg = cv2.imread(FG_PATH)
bg = cv2.imread(BG_PATH)
fg2 = cv2.imread(FG2_PATH)
bg_indoor = cv2.imread(BG_INDOOR)
bg_night = cv2.imread(BG_NIGHT)

# 统一缩放前景到合适大小方便显示
fg_small = cv2.resize(fg, (240, 320))
bg_small = cv2.resize(bg, (480, 320))

# 步骤1: 原图
save(fg_small, 'process_01_fg_original.png')

# 步骤2: 灰度化
fg_gray = cv2.cvtColor(fg_small, cv2.COLOR_BGR2GRAY)
fg_gray_bgr = gray_to_bgr(fg_gray)
save(fg_gray_bgr, 'process_02_fg_gray.png')

# 步骤3: 高斯滤波(背景)
bg_gauss = Preprocessor.denoise(bg_small, 'gaussian')
save(bg_gauss, 'process_03_bg_gaussian.png')

# 步骤4: 双边滤波(前景)
fg_bilateral = Preprocessor.denoise(fg_small, 'bilateral')
save(fg_bilateral, 'process_04_fg_bilateral.png')

# 步骤5: 去噪对比（高斯 vs 双边）
denoise_compare = np.hstack([
    cv2.resize(fg_bilateral, (200, 260)),
    cv2.resize(fg_gray_bgr, (200, 260)),
    cv2.resize(gray_to_bgr(cv2.medianBlur(fg_gray, 3)), (200, 260))
])
put_text(denoise_compare, '双边滤波(保留边缘)', (8, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(denoise_compare, '灰度原图', (208, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(denoise_compare, '中值滤波', (408, 22), 0.5, (255,255,255), 1, (0,0,0))
save(denoise_compare, 'process_05_denoise_compare.png')

# 步骤6: CLAHE增强
fg_clahe = Preprocessor.equalize_hist(fg_bilateral)
save(fg_clahe, 'process_06_fg_clahe.png')

# 步骤7: 增强对比（原始去噪 vs CLAHE）
enhance_compare = np.hstack([
    cv2.resize(fg_small, (220, 280)),
    cv2.resize(fg_bilateral, (220, 280)),
    cv2.resize(fg_clahe, (220, 280))
])
put_text(enhance_compare, '原图', (8, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(enhance_compare, '双边滤波后', (228, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(enhance_compare, 'CLAHE增强后', (448, 22), 0.5, (255,255,255), 1, (0,0,0))
save(enhance_compare, 'process_07_enhance_compare.png')

# ── 3.3.1 GrabCut ──
print('\n[3.3.1] GrabCut前景提取')
fg_denoised = Preprocessor.denoise(fg, 'bilateral')
fg_denoised_small = cv2.resize(fg_denoised, (240, 320))

# GrabCut流程
# 手动一步一步执行GrabCut以便截取中间状态
h, w = fg_denoised_small.shape[:2]
mask_gc = np.zeros((h, w), np.uint8)
bgd_model = np.zeros((1, 65), np.float64)
fgd_model = np.zeros((1, 65), np.float64)
margin = int(min(h, w) * 0.1)
rect = (margin, margin, w - 2 * margin, h - 2 * margin)

# 初始化矩形示意
rect_viz = fg_denoised_small.copy()
cv2.rectangle(rect_viz, (margin, margin), (w - margin, h - margin), (0, 255, 0), 3)
put_text(rect_viz, 'GrabCut初始化矩形', (8, 22), 0.6, (0,255,0), 2, (0,0,0))
save(rect_viz, 'process_08_grabcut_rect.png')

# 迭代1次后的原始GrabCut掩码（仅矩形初始化）
cv2.grabCut(fg_denoised_small, mask_gc, rect, bgd_model, fgd_model, 1, cv2.GC_INIT_WITH_RECT)
mask_raw_viz = np.where(
    (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
    255, 0
).astype(np.uint8)
save(gray_to_bgr(mask_raw_viz), 'process_09_grabcut_raw_1iter.png')

# 迭代5次结果
cv2.grabCut(fg_denoised_small, mask_gc, rect, bgd_model, fgd_model, 4, cv2.GC_INIT_WITH_RECT)
mask_raw_final = np.where(
    (mask_gc == cv2.GC_FGD) | (mask_gc == cv2.GC_PR_FGD),
    255, 0
).astype(np.uint8)
save(gray_to_bgr(mask_raw_final), 'process_10_grabcut_raw_5iter.png')

# 形态学闭运算
kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
mask_close = cv2.morphologyEx(mask_raw_final, cv2.MORPH_CLOSE, kernel_5, iterations=3)
save(gray_to_bgr(mask_close), 'process_11_grabcut_morph_close.png')

# 形态学开运算
mask_open = cv2.morphologyEx(mask_close, cv2.MORPH_OPEN, kernel_5, iterations=1)
save(gray_to_bgr(mask_open), 'process_12_grabcut_morph_open.png')

mask_grabcut_final = mask_open

# GrabCut最终叠加预览
preview_gc = fg_denoised_small.copy()
preview_gc[mask_grabcut_final == 0] = (preview_gc[mask_grabcut_final == 0] * 0.25).astype(np.uint8)
save(preview_gc, 'process_13_grabcut_preview.png')

# ── 3.3.2 颜色阈值 ──
print('\n[3.3.2] 颜色阈值分割')
# HSV色相通道可视化
hsv_fg = cv2.cvtColor(fg_denoised_small, cv2.COLOR_BGR2HSV)
h_ch, s_ch, v_ch = cv2.split(hsv_fg)
h_norm = cv2.normalize(h_ch, None, 0, 255, cv2.NORM_MINMAX)
save(gray_to_bgr(h_norm), 'process_14_hsv_hue_channel.png')

# HSV颜色范围可视化——选取绿幕范围
hsv_display = fg_denoised_small.copy()
# 在HSV空间高亮绿幕区域(绿色范围)
lo = np.array([35, 40, 40])
hi = np.array([85, 255, 255])
green_mask = cv2.inRange(hsv_fg, lo, hi)
hsv_highlight = fg_denoised_small.copy()
hsv_highlight[green_mask > 0] = (0, 255, 0)  # 绿色高亮背景区域
put_text(hsv_highlight, 'HSV绿色范围检测(背景)', (8, 22), 0.55, (0,255,0), 2, (0,0,0))
save(hsv_highlight, 'process_15_hsv_green_detection.png')

# 取反得到前景mask
mask_hsv = cv2.bitwise_not(green_mask)
# 形态学后处理
kernel_7 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
mask_hsv_close = cv2.morphologyEx(mask_hsv, cv2.MORPH_CLOSE, kernel_7, iterations=2)
mask_hsv_open = cv2.morphologyEx(mask_hsv_close, cv2.MORPH_OPEN, kernel_7, iterations=1)
save(gray_to_bgr(mask_hsv_open), 'process_16_hsv_final_mask.png')

# 叠加预览
preview_hsv = fg_denoised_small.copy()
preview_hsv[mask_hsv_open == 0] = (preview_hsv[mask_hsv_open == 0] * 0.25).astype(np.uint8)
save(preview_hsv, 'process_17_hsv_preview.png')

# ── 3.3.3 Canny+形态学 ──
print('\n[3.3.3] Canny+形态学')
gray_canny = cv2.cvtColor(fg_denoised_small, cv2.COLOR_BGR2GRAY)
save(gray_to_bgr(gray_canny), 'process_18_canny_gray.png')

blur_canny = cv2.GaussianBlur(gray_canny, (5, 5), 0)
save(gray_to_bgr(blur_canny), 'process_19_canny_gaussian.png')

edges = cv2.Canny(blur_canny, 50, 150)
save(gray_to_bgr(edges), 'process_20_canny_edges.png')

kernel_5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated = cv2.dilate(edges, kernel_5, iterations=2)
save(gray_to_bgr(dilated), 'process_21_canny_dilate.png')

# 轮廓查找
contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
contour_viz = fg_denoised_small.copy()
cv2.drawContours(contour_viz, contours, -1, (0, 255, 0), 2)
put_text(contour_viz, f'找到{len(contours)}个轮廓', (8, 22), 0.55, (0,255,0), 2, (0,0,0))
save(contour_viz, 'process_22_canny_contours.png')

# 填充
mask_canny = np.zeros(fg_denoised_small.shape[:2], np.uint8)
contours_sorted = sorted(contours, key=cv2.contourArea, reverse=True)
for cnt in contours_sorted[:3]:
    if cv2.contourArea(cnt) > 500:
        cv2.drawContours(mask_canny, [cnt], -1, 255, -1)

mask_canny_close = cv2.morphologyEx(mask_canny, cv2.MORPH_CLOSE, kernel_5, iterations=3)
save(gray_to_bgr(mask_canny_close), 'process_23_canny_final_mask.png')

# 叠加预览
preview_canny = fg_denoised_small.copy()
preview_canny[mask_canny_close == 0] = (preview_canny[mask_canny_close == 0] * 0.25).astype(np.uint8)
save(preview_canny, 'process_24_canny_preview.png')

# 三种方法对比
three_methods = make_compare_img(
    [preview_gc, preview_hsv, preview_canny],
    ['GrabCut', 'HSV颜色阈值', 'Canny+形态学'],
    target_h=280
)
save(three_methods, 'process_25_three_methods_compare.png')

# ── 3.3.4 边缘羽化 ──
print('\n[3.3.4] 边缘羽化')

# 硬掩码局部放大（边缘锯齿）
hard_crop = cv2.resize(mask_grabcut_final, (240, 320))
hard_edge = hard_crop.copy()
# 在边缘区域放大
edge_region_hard = hard_crop[130:190, 80:180]
edge_region_hard_bgr = gray_to_bgr(edge_region_hard)
edge_region_hard_big = cv2.resize(edge_region_hard_bgr, (200, 200), interpolation=cv2.INTER_NEAREST)
# 画网格线显示锯齿
for i in range(0, 201, 20):
    cv2.line(edge_region_hard_big, (i, 0), (i, 200), (100, 100, 100), 1)
    cv2.line(edge_region_hard_big, (0, i), (200, i), (100, 100, 100), 1)

# 羽化掩码
mask_float = mask_grabcut_final.astype(np.float32) / 255.0
feather_s3 = cv2.GaussianBlur(mask_float, (21, 21), sigmaX=3)
feather_s5 = cv2.GaussianBlur(mask_float, (21, 21), sigmaX=5)
feather_s9 = cv2.GaussianBlur(mask_float, (21, 21), sigmaX=9)

feather_vis_s3 = (feather_s3 * 255).astype(np.uint8)
feather_vis_s5 = (feather_s5 * 255).astype(np.uint8)
feather_vis_s9 = (feather_s9 * 255).astype(np.uint8)

save(gray_to_bgr(feather_vis_s3), 'process_26_feather_sigma3.png')
save(gray_to_bgr(feather_vis_s5), 'process_27_feather_sigma5.png')
save(gray_to_bgr(feather_vis_s9), 'process_28_feather_sigma9.png')

# 硬vs软对比（边缘局部放大）
edge_region_soft = feather_vis_s5[130:190, 80:180]
edge_region_soft_big = cv2.resize(gray_to_bgr(edge_region_soft), (200, 200), interpolation=cv2.INTER_NEAREST)
for i in range(0, 201, 20):
    cv2.line(edge_region_soft_big, (i, 0), (i, 200), (100, 100, 100), 1)
    cv2.line(edge_region_soft_big, (0, i), (200, i), (100, 100, 100), 1)

# 完整掩码对比
hard_full = cv2.resize(gray_to_bgr(mask_grabcut_final), (200, 200))
soft_full = cv2.resize(gray_to_bgr(feather_vis_s5), (200, 200))
mask_compare = np.hstack([
    hard_full,
    soft_full,
    edge_region_hard_big,
    edge_region_soft_big
])
put_text(mask_compare, '二值硬掩码', (8, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(mask_compare, '羽化软掩码(sigma=5)', (208, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(mask_compare, '硬边缘(局部放大)', (408, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(mask_compare, '软边缘(局部放大)', (608, 22), 0.5, (255,255,255), 1, (0,0,0))
save(mask_compare, 'process_29_feather_compare.png')

# ── 3.4.1 拉普拉斯金字塔融合 ──
print('\n[3.4.1] 拉普拉斯金字塔融合')
mask_refined = ForegroundExtractor.refine_mask(mask_grabcut_final)
pos = (50, 50)

fg_resized = Preprocessor.resize_to_match(fg_denoised_small, bg_small)
mask_f = cv2.resize(mask_refined, (fg_resized.shape[1], fg_resized.shape[0]))

# ROI区域
x, y = pos
fh, fw = fg_resized.shape[:2]
bh, bw = bg_small.shape[:2]
x1, y1 = max(0, x), max(0, y)
x2 = min(bw, x + fw); y2 = min(bh, y + fh)
fx1 = x1 - x; fy1 = y1 - y
fx2 = fx1 + (x2 - x1); fy2 = fy1 + (y2 - y1)

fg_roi = fg_resized[fy1:fy2, fx1:fx2]
bg_roi = bg_small[y1:y2, x1:x2].copy()
mask_roi = mask_f[fy1:fy2, fx1:fx2]

# 用虚线框标记合成位置
pos_viz = bg_small.copy()
cv2.rectangle(pos_viz, (x1, y1), (x2, y2), (0, 255, 255), 2)
put_text(pos_viz, '合成位置(ROI)', (x1+5, y1-8), 0.5, (0,255,255), 1, (0,0,0))
save(pos_viz, 'process_30_lap_position.png')

# 前景ROI、背景ROI、掩码ROI
save(cv2.resize(fg_roi, (200, 260)), 'process_31_lap_fg_roi.png')
save(cv2.resize(bg_roi, (200, 260)), 'process_32_lap_bg_roi.png')
save(cv2.resize(gray_to_bgr((mask_roi*255).astype(np.uint8)), (200, 260)), 'process_33_lap_mask_roi.png')

# 高斯金字塔各层
mask_roi_3d = mask_roi[..., np.newaxis] if mask_roi.ndim == 2 else mask_roi
gp_fg = [fg_roi.astype(np.float32)]
gp_bg = [bg_roi.astype(np.float32)]
gp_mask = [mask_roi_3d.astype(np.float32)]
for _ in range(3):
    gp_fg.append(cv2.pyrDown(gp_fg[-1]))
    gp_bg.append(cv2.pyrDown(gp_bg[-1]))
    gp_mask.append(cv2.pyrDown(gp_mask[-1]))

lap_pyramid_viz = []
for level in range(3):
    up = cv2.pyrUp(gp_fg[level+1], dstsize=(gp_fg[level].shape[1], gp_fg[level].shape[0]))
    lap = gp_fg[level] - up  # 拉普拉斯系数（高频细节）
    lap_viz = cv2.normalize(lap, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    lap_viz_r = cv2.resize(lap_viz, (180, 140))
    # 高斯低频
    low_viz = cv2.resize(gp_fg[level].astype(np.uint8), (180, 140))
    # 融合权重
    m_raw = cv2.resize((gp_mask[level]*255).astype(np.uint8), (180, 140))
    if m_raw.ndim == 3:
        m_raw = m_raw[:,:,0]
    m_viz = cv2.cvtColor(m_raw, cv2.COLOR_GRAY2BGR)
    row = np.hstack([low_viz, lap_viz_r, m_viz])
    put_text(row, f'Level {level} 低频', (8, 18), 0.45, (255,255,255), 1, (0,0,0))
    put_text(row, f'Level {level} 高频', (188, 18), 0.45, (255,255,255), 1, (0,0,0))
    put_text(row, f'Level {level} 权重', (368, 18), 0.45, (255,255,255), 1, (0,0,0))
    lap_pyramid_viz.append(row)

pyramid_display = np.vstack(lap_pyramid_viz)
save(pyramid_display, 'process_34_lap_pyramid_levels.png')

# 拉普拉斯金字塔重建结果
res_laplacian = ImageCompositor.laplacian_pyramid_blend(fg_resized, bg_small, mask_roi_3d, pos)
save(res_laplacian, 'process_35_lap_result.png')

# 对比：Alpha混合 vs 拉普拉斯
res_alpha = ImageCompositor.alpha_blend(fg_resized, bg_small, mask_roi, pos)
blend_compare = np.hstack([
    cv2.resize(res_alpha, (260, 320)),
    cv2.resize(res_laplacian, (260, 320))
])
put_text(blend_compare, 'Alpha混合（边缘锯齿）', (8, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(blend_compare, '拉普拉斯金字塔融合（边缘自然）', (268, 22), 0.55, (255,255,255), 2, (0,0,0))
save(blend_compare, 'process_36_alpha_vs_laplacian.png')

# ── 3.4.2 颜色匹配融合 ──
print('\n[3.4.2] 颜色匹配融合')
res_color = ImageCompositor.color_match_blend(fg_resized, bg_small, mask_roi_3d, pos)

# 颜色迁移前后对比
# 手动模拟：在LAB空间做颜色迁移
fg_roi_lab = cv2.cvtColor(fg_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
bg_roi_lab = cv2.cvtColor(bg_roi, cv2.COLOR_BGR2LAB).astype(np.float32)
fg_matched = fg_roi_lab.copy()
for c in range(3):
    fg_mean, fg_std = fg_roi_lab[:,:,c].mean(), fg_roi_lab[:,:,c].std() + 1e-6
    bg_mean, bg_std = bg_roi_lab[:,:,c].mean(), bg_roi_lab[:,:,c].std() + 1e-6
    fg_matched[:,:,c] = (fg_roi_lab[:,:,c] - fg_mean) * (bg_std / fg_std) + bg_mean
fg_matched_bgr = cv2.cvtColor(np.clip(fg_matched, 0, 255).astype(np.uint8), cv2.COLOR_LAB2BGR)

color_compare = np.hstack([
    cv2.resize(fg_roi, (200, 260)),
    cv2.resize(bg_roi, (200, 260)),
    cv2.resize(fg_matched_bgr, (200, 260)),
    cv2.resize(res_color, (200, 260))
])
put_text(color_compare, '前景ROI(原色)', (8, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(color_compare, '背景ROI(目标色调)', (208, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(color_compare, '颜色迁移后前景', (408, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(color_compare, '融合结果', (608, 22), 0.5, (255,255,255), 1, (0,0,0))
save(color_compare, 'process_37_color_match_process.png')

# ── 3.4.3 泊松融合 ──
print('\n[3.4.3] 泊松融合')
mask_binary = cv2.resize(mask_grabcut_final, (fg_resized.shape[1], fg_resized.shape[0]))
mask_binary_roi = mask_binary[fy1:fy2, fx1:fx2]
res_poisson = ImageCompositor.poisson_blend(fg_resized, bg_small, mask_binary, pos)

# 泊松融合过程示意
poisson_viz = bg_small.copy()
# 标记前景插入区域
cv2.rectangle(poisson_viz, (x1, y1), (x2, y2), (0, 0, 255), 2)
put_text(poisson_viz, '泊松求解区域', (x1+5, y1-8), 0.5, (0,0,255), 1, (0,0,0))

poisson_compare = np.hstack([
    cv2.resize(gray_to_bgr(mask_binary_roi), (200, 260)),
    cv2.resize(poisson_viz, (200, 260)),
    cv2.resize(bg_roi, (200, 260)),
    cv2.resize(res_poisson, (200, 260))
])
put_text(poisson_compare, '掩码ROI', (8, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(poisson_compare, '梯度场约束区域', (208, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(poisson_compare, '背景ROI(边界条件)', (408, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(poisson_compare, '泊松融合结果', (608, 22), 0.5, (255,255,255), 1, (0,0,0))
save(poisson_compare, 'process_38_poisson_process.png')

# 四种合成方法并列对比
four_methods = hstack_resize([
    res_alpha, res_laplacian, res_color, res_poisson
], 280)
put_text(four_methods, 'Alpha混合', (10, 25), 0.6, (255,255,255), 2, (0,0,0))
put_text(four_methods, '拉普拉斯金字塔', (four_methods.shape[1]//4+10, 25), 0.6, (255,255,255), 2, (0,0,0))
put_text(four_methods, '颜色匹配融合', (four_methods.shape[1]//2+10, 25), 0.6, (255,255,255), 2, (0,0,0))
put_text(four_methods, '泊松融合', (3*four_methods.shape[1]//4+10, 25), 0.6, (255,255,255), 2, (0,0,0))
save(four_methods, 'process_39_four_methods_compare.png')

# ── 3.5 后处理增强（保留原有风格，命名改为process_） ──
print('\n[3.5] 后处理增强')
# 锐化对比
sharp_0 = PostProcessor.sharpen(res_laplacian, 0)
sharp_1 = PostProcessor.sharpen(res_laplacian, 1.0)
sharp_2 = PostProcessor.sharpen(res_laplacian, 2.0)
sharp_compare = make_compare_img(
    [sharp_0, sharp_1, sharp_2],
    ['strength=0', 'strength=1.0', 'strength=2.0'],
    240, 0.55
)
save(sharp_compare, 'process_40_post_sharpen.png')

# 对比度/亮度对比
contrast_low = PostProcessor.adjust_contrast_brightness(res_laplacian, 0.5, 0)
contrast_high = PostProcessor.adjust_contrast_brightness(res_laplacian, 1.5, 0)
bright_high = PostProcessor.adjust_contrast_brightness(res_laplacian, 1.0, 30)
contrast_compare = make_compare_img(
    [contrast_low, res_laplacian, contrast_high, bright_high],
    ['alpha=0.5', 'alpha=1.0', 'alpha=1.5', 'beta=30'],
    200, 0.45
)
save(contrast_compare, 'process_41_post_contrast.png')

# 饱和度对比
sat_low = PostProcessor.color_adjust(res_laplacian, sat_scale=0.3)
sat_high = PostProcessor.color_adjust(res_laplacian, sat_scale=1.8)
sat_compare = make_compare_img(
    [sat_low, res_laplacian, sat_high],
    ['sat=0.3', 'sat=1.0', 'sat=1.8'],
    240, 0.55
)
save(sat_compare, 'process_42_post_saturation.png')

# 暗角效果对比
vig_0 = PostProcessor.vignette(res_laplacian, 0)
vig_1 = PostProcessor.vignette(res_laplacian, 0.4)
vig_2 = PostProcessor.vignette(res_laplacian, 0.8)
vig_compare = make_compare_img(
    [vig_0, vig_1, vig_2],
    ['strength=0', 'strength=0.4', 'strength=0.8'],
    240, 0.55
)
save(vig_compare, 'process_43_post_vignette.png')

# ── 3.6 流程总览（收集所有步骤缩略图） ──
print('\n[3.6] 处理流程总览')
# 收集已有process图
step_previews = [
    ('process_01_fg_original.png', '原图'),
    ('process_02_fg_gray.png', '灰度化'),
    ('process_04_fg_bilateral.png', '双边滤波'),
    ('process_06_fg_clahe.png', 'CLAHE增强'),
    ('process_08_grabcut_rect.png', 'GrabCut初始化'),
    ('process_10_grabcut_raw_5iter.png', 'GrabCut分割'),
    ('process_11_grabcut_morph_close.png', '闭运算'),
    ('process_12_grabcut_morph_open.png', '开运算'),
    ('process_27_feather_sigma5.png', '边缘羽化'),
    ('process_31_lap_fg_roi.png', '前景ROI'),
    ('process_32_lap_bg_roi.png', '背景ROI'),
    ('process_35_lap_result.png', '合成结果'),
]
available = []
for fname, label in step_previews:
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        img = cv2.imread(path)
        if img is not None:
            img_r = cv2.resize(img, (150, 120))
            put_text(img_r, label, (5, 18), 0.4, (255,255,255), 1, None)
            available.append(img_r)

if len(available) >= 8:
    rows = []
    for i in range(0, len(available), 6):
        row = np.hstack(available[i:i+6])
        rows.append(row)
    max_w = max(r.shape[1] for r in rows)
    for i in range(len(rows)):
        if rows[i].shape[1] < max_w:
            rows[i] = cv2.copyMakeBorder(rows[i], 0, 0, 0, max_w - rows[i].shape[1],
                                          cv2.BORDER_CONSTANT, value=(255,255,255))
    pipeline = np.vstack(rows)
    save(pipeline, 'process_44_pipeline_overview.png')

# ── 4.2 多场景应用 ──
print('\n[4.2] 多场景应用')

# 场景1：室内背景（展示全过程）
fg_s1 = Preprocessor.resize_to_match(fg_denoised, bg_indoor)
m_s1 = cv2.resize(mask_refined, (fg_s1.shape[1], fg_s1.shape[0]))
m_s1_3d = m_s1[..., np.newaxis] if m_s1.ndim == 2 else m_s1

save(cv2.resize(fg_s1, (200, 260)), 'process_45_scene1_fg.png')

# GrabCut掩码（对于室内场景重新跑GrabCut）
m_s1_bin = ForegroundExtractor.grabcut(fg_s1)[0]
m_s1_refined = ForegroundExtractor.refine_mask(m_s1_bin)
save(gray_to_bgr(m_s1_bin), 'process_46_scene1_mask.png')

result_s1 = ImageCompositor.laplacian_pyramid_blend(fg_s1, bg_indoor, m_s1_3d, pos)
save(result_s1, 'process_47_scene1_result.png')

# 场景1：每种方法对比
m_s1_b = cv2.resize(m_s1_bin, (fg_s1.shape[1], fg_s1.shape[0]))
s1_alpha = ImageCompositor.alpha_blend(fg_s1, bg_indoor, m_s1_refined, pos)
s1_lap   = ImageCompositor.laplacian_pyramid_blend(fg_s1, bg_indoor, m_s1_3d, pos)
s1_color = ImageCompositor.color_match_blend(fg_s1, bg_indoor, m_s1_3d, pos)
s1_poisson = ImageCompositor.poisson_blend(fg_s1, bg_indoor, m_s1_b, pos)

s1_four = hstack_resize([s1_alpha, s1_lap, s1_color, s1_poisson], 240)
put_text(s1_four, 'Alpha', (10, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(s1_four, 'Laplacian', (s1_four.shape[1]//4+10, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(s1_four, 'ColorMatch', (s1_four.shape[1]//2+10, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(s1_four, 'Poisson', (3*s1_four.shape[1]//4+10, 22), 0.55, (255,255,255), 2, (0,0,0))
save(s1_four, 'process_48_scene1_four_methods.png')

# 场景2：夜景背景
fg_s2 = Preprocessor.resize_to_match(fg_denoised, bg_night)
m_s2 = cv2.resize(mask_refined, (fg_s2.shape[1], fg_s2.shape[0]))
m_s2_3d = m_s2[..., np.newaxis] if m_s2.ndim == 2 else m_s2
result_s2 = ImageCompositor.laplacian_pyramid_blend(fg_s2, bg_night, m_s2_3d, pos)
save(result_s2, 'process_49_scene2_night_result.png')

# 场景2：四种方法对比
m_s2_b = cv2.resize(mask_grabcut_final, (fg_s2.shape[1], fg_s2.shape[0]))
s2_alpha = ImageCompositor.alpha_blend(fg_s2, bg_night, m_s2, pos)
s2_lap   = ImageCompositor.laplacian_pyramid_blend(fg_s2, bg_night, m_s2_3d, pos)
s2_color = ImageCompositor.color_match_blend(fg_s2, bg_night, m_s2_3d, pos)
s2_poisson = ImageCompositor.poisson_blend(fg_s2, bg_night, m_s2_b, pos)

s2_four = hstack_resize([s2_alpha, s2_lap, s2_color, s2_poisson], 240)
put_text(s2_four, 'Alpha', (10, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(s2_four, 'Laplacian', (s2_four.shape[1]//4+10, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(s2_four, 'ColorMatch', (s2_four.shape[1]//2+10, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(s2_four, 'Poisson', (3*s2_four.shape[1]//4+10, 22), 0.55, (255,255,255), 2, (0,0,0))
save(s2_four, 'process_50_scene2_four_methods.png')

# 场景3：花朵前景
fg2_denoised = Preprocessor.denoise(fg2, 'bilateral')
fg2_s = Preprocessor.resize_to_match(fg2_denoised, bg_small)
save(cv2.resize(fg2_s, (200, 260)), 'process_51_scene3_fg_flower.png')

mask_f2_bin = ForegroundExtractor.color_threshold(fg2_denoised, 'white')
m_f2 = ForegroundExtractor.refine_mask(mask_f2_bin)
save(gray_to_bgr(mask_f2_bin), 'process_52_scene3_mask_flower.png')

m_s3 = cv2.resize(m_f2, (fg2_s.shape[1], fg2_s.shape[0]))
m_s3_3d = m_s3[..., np.newaxis] if m_s3.ndim == 2 else m_s3
result_s3 = ImageCompositor.laplacian_pyramid_blend(fg2_s, bg_small, m_s3_3d, pos)
save(result_s3, 'process_53_scene3_result_flower.png')

# 场景3：四种方法对比
m_s3_b = cv2.resize(mask_f2_bin, (fg2_s.shape[1], fg2_s.shape[0]))
s3_alpha = ImageCompositor.alpha_blend(fg2_s, bg_small, m_s3, pos)
s3_lap   = ImageCompositor.laplacian_pyramid_blend(fg2_s, bg_small, m_s3_3d, pos)
s3_color = ImageCompositor.color_match_blend(fg2_s, bg_small, m_s3_3d, pos)
s3_poisson = ImageCompositor.poisson_blend(fg2_s, bg_small, m_s3_b, pos)

s3_four = hstack_resize([s3_alpha, s3_lap, s3_color, s3_poisson], 240)
put_text(s3_four, 'Alpha', (10, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(s3_four, 'Laplacian', (s3_four.shape[1]//4+10, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(s3_four, 'ColorMatch', (s3_four.shape[1]//2+10, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(s3_four, 'Poisson', (3*s3_four.shape[1]//4+10, 22), 0.55, (255,255,255), 2, (0,0,0))
save(s3_four, 'process_54_scene3_four_methods.png')

# 三个场景最终结果并列
three_scenes = hstack_resize([result_s1, result_s2, result_s3], 300)
put_text(three_scenes, '场景1: 室外→室内', (8, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(three_scenes, '场景2: 室外→夜景', (three_scenes.shape[1]//3+8, 22), 0.55, (255,255,255), 2, (0,0,0))
put_text(three_scenes, '场景3: 花朵→天空', (2*three_scenes.shape[1]//3+8, 22), 0.55, (255,255,255), 2, (0,0,0))
save(three_scenes, 'process_55_three_scenes_compare.png')

# 失败案例：前景偏离中心
print('\n[4.3] 失败案例分析')
# 模拟偏中心前景：裁剪左半部分
h_f, w_f = fg_denoised.shape[:2]
fg_offset = fg_denoised[:, w_f//2:]  # 只取右半部分（人物偏右）
fg_offset = cv2.resize(fg_offset, (240, 320))
mask_off_bin = ForegroundExtractor.grabcut(fg_offset)[0]
save(gray_to_bgr(mask_off_bin), 'process_56_failure_offset_grabcut.png')

# 叠加预览
preview_off = fg_offset.copy()
preview_off[mask_off_bin == 0] = (preview_off[mask_off_bin == 0] * 0.25).astype(np.uint8)
save(preview_off, 'process_57_failure_offset_preview.png')

# 正常GrabCut vs 偏中心对比
failure_compare = np.hstack([
    cv2.resize(preview_gc, (200, 260)),
    cv2.resize(preview_off, (200, 260))
])
put_text(failure_compare, '正常GrabCut（目标居中）', (8, 22), 0.5, (255,255,255), 1, (0,0,0))
put_text(failure_compare, '偏中心GrabCut（分割不完整）', (208, 22), 0.5, (255,255,255), 1, (0,0,0))
save(failure_compare, 'process_58_failure_compare.png')

print(f'\n所有图片已保存至: {OUT}')
import glob
pngs = [f for f in os.listdir(OUT) if f.startswith('process_') and f.endswith('.png')]
print(f'共生成 {len(pngs)} 张 process_ 图片')
