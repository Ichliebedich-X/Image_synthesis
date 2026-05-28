"""
为实验报告生成所有中间结果图片。
Run: python generate_report_images.py
输出到 D:/report_images/ 目录。
"""
import sys
sys.path.insert(0, 'D:\\AiStdio\\Python\\PythonProject\\DIP')

import cv2
import numpy as np
from PIL import Image
import os
from Image_synthesis import (
    Preprocessor, ForegroundExtractor, ImageCompositor, PostProcessor
)

OUT = r'D:\report_images'
os.makedirs(OUT, exist_ok=True)

FG_PATH = r'D:\AiStdio\Python\PythonProject\DIP\dataset\foreground\person_greenscreen.jpg'
BG_PATH = r'D:\AiStdio\Python\PythonProject\DIP\dataset\background\outdoor_sky.jpg'
BG2_PATH = r'D:\AiStdio\Python\PythonProject\DIP\dataset\background\night_gradient.jpg'
FG2_PATH = r'D:\AiStdio\Python\PythonProject\DIP\dataset\foreground\flower_whitebg.jpg'

def save(img, name):
    """保存图像到输出目录"""
    path = os.path.join(OUT, name)
    cv2.imwrite(path, img)
    print(f'  Saved: {name} ({img.shape[1]}x{img.shape[0]})')
    return path

def img_to_pil(img):
    rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return Image.fromarray(rgb)

print('='*60)
print('生成实验报告所需图片...')
print('='*60)

# ── 1. 加载测试图像 ──
print('\n[1] 加载测试图像')
fg = cv2.imread(FG_PATH)
bg = cv2.imread(BG_PATH)
fg2 = cv2.imread(FG2_PATH)
bg2 = cv2.imread(BG2_PATH)
save(fg, '01_fg_original.png')
save(bg, '02_bg_original.png')

# ── 2. 预处理 ──
print('\n[2] 图像预处理')
steps = Preprocessor.get_steps(fg, bg)
step_names = {
    '前景原图': 'fg_original', '前景去噪(双边)': 'fg_denoised_bilateral',
    '前景增强(CLAHE)': 'fg_clahe', '背景原图': 'bg_original',
    '背景去噪(高斯)': 'bg_denoised_gaussian',
}
for title, img in steps.items():
    safe_name = step_names.get(title, title.replace('(', '_').replace(')', '').replace(' ', '_'))
    save(img, f'03_prep_{safe_name}.png')

# 单独保存灰度图用于报告
gray = cv2.cvtColor(fg, cv2.COLOR_BGR2GRAY)
gray_bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
save(gray_bgr, '03b_fg_gray.png')

# ── 3. 前景提取 ──
print('\n[3] 前景提取')
fg_denoised = Preprocessor.denoise(fg, 'bilateral')
save(fg_denoised, '04_fg_denoised.png')

# 3a. GrabCut
mask_grabcut_bin, mask_grabcut_raw = ForegroundExtractor.grabcut(fg_denoised)
mask_grabcut_vis = cv2.cvtColor(mask_grabcut_bin, cv2.COLOR_GRAY2BGR)
save(mask_grabcut_vis, '05_mask_grabcut.png')

# GrabCut 预览
preview_grabcut = fg.copy()
preview_grabcut[mask_grabcut_bin == 0] = (preview_grabcut[mask_grabcut_bin == 0] * 0.3).astype(np.uint8)
save(preview_grabcut, '05b_preview_grabcut.png')

# 形态学前后的GrabCut对比
mask_before_morph = np.where(
    (mask_grabcut_raw == cv2.GC_FGD) | (mask_grabcut_raw == cv2.GC_PR_FGD),
    255, 0
).astype(np.uint8)
mask_before_morph = cv2.cvtColor(mask_before_morph, cv2.COLOR_GRAY2BGR)
save(mask_before_morph, '05c_mask_grabcut_before_morph.png')

# 3b. HSV颜色阈值
mask_hsv_bin = ForegroundExtractor.color_threshold(fg_denoised, 'green')
mask_hsv_vis = cv2.cvtColor(mask_hsv_bin, cv2.COLOR_GRAY2BGR)
save(mask_hsv_vis, '06_mask_hsv_green.png')

preview_hsv = fg.copy()
preview_hsv[mask_hsv_bin == 0] = (preview_hsv[mask_hsv_bin == 0] * 0.3).astype(np.uint8)
save(preview_hsv, '06b_preview_hsv.png')

# 3c. Canny+形态学
mask_canny_bin = ForegroundExtractor.canny_morphology(fg_denoised)
mask_canny_vis = cv2.cvtColor(mask_canny_bin, cv2.COLOR_GRAY2BGR)
save(mask_canny_vis, '07_mask_canny.png')

preview_canny = fg.copy()
preview_canny[mask_canny_bin == 0] = (preview_canny[mask_canny_bin == 0] * 0.3).astype(np.uint8)
save(preview_canny, '07b_preview_canny.png')

# Canny中间步骤
gray_canny = cv2.cvtColor(fg_denoised, cv2.COLOR_BGR2GRAY)
blurred = cv2.GaussianBlur(gray_canny, (5, 5), 0)
edges = cv2.Canny(blurred, 50, 150)
edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
save(edges_bgr, '07c_canny_edges.png')
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
dilated = cv2.dilate(edges, kernel, iterations=2)
dilated_bgr = cv2.cvtColor(dilated, cv2.COLOR_GRAY2BGR)
save(dilated_bgr, '07d_canny_dilated.png')

# 3d. 边缘羽化
mask_refined = ForegroundExtractor.refine_mask(mask_grabcut_bin)
mask_refined_vis = (mask_refined * 255).astype(np.uint8)
mask_refined_bgr = cv2.cvtColor(mask_refined_vis, cv2.COLOR_GRAY2BGR)
save(mask_refined_bgr, '08_mask_refined.png')
# 硬掩码vs软掩码对比
mask_hard_soft = np.hstack([
    cv2.cvtColor(mask_grabcut_bin, cv2.COLOR_GRAY2BGR),
    mask_refined_bgr
])
save(mask_hard_soft, '08b_mask_hard_vs_soft.png')

# ── 4. 图像合成 ──
print('\n[4] 图像合成')
fg_resized = Preprocessor.resize_to_match(fg_denoised, bg)
mh, mw = mask_grabcut_bin.shape[:2]
fh, fw = fg_resized.shape[:2]
mask_b = cv2.resize(mask_grabcut_bin, (fw, fh))
mask_f = cv2.resize(mask_refined, (fw, fh))
pos = (50, 50)

# 4a. Alpha混合
res_alpha = ImageCompositor.alpha_blend(fg_resized, bg, mask_f, pos)
save(res_alpha, '09_result_alpha.png')

# 4b. 拉普拉斯金字塔融合
# 确保 mask 是 3D 以适配广播
mask_f_3d = mask_f[..., np.newaxis] if mask_f.ndim == 2 else mask_f
res_laplacian = ImageCompositor.laplacian_pyramid_blend(fg_resized, bg, mask_f_3d, pos)
save(res_laplacian, '10_result_laplacian.png')

# 4c. 颜色匹配融合
res_color = ImageCompositor.color_match_blend(fg_resized, bg, mask_f_3d, pos)
save(res_color, '11_result_color_match.png')

# 4d. 泊松融合
res_poisson = ImageCompositor.poisson_blend(fg_resized, bg, mask_b, pos)
save(res_poisson, '12_result_poisson.png')

# 四种方法并列对比
four_methods = np.hstack([
    cv2.resize(res_alpha, (300, 240)),
    cv2.resize(res_laplacian, (300, 240)),
    cv2.resize(res_color, (300, 240)),
    cv2.resize(res_poisson, (300, 240)),
])
# 加文字标注
h, w = four_methods.shape[:2]
cv2.putText(four_methods, 'Alpha', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
cv2.putText(four_methods, 'Laplacian', (310, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
cv2.putText(four_methods, 'ColorMatch', (610, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
cv2.putText(four_methods, 'Poisson', (910, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255,255,255), 2)
save(four_methods, '13_four_methods_comparison.png')

# ── 5. 后处理增强 ──
print('\n[5] 后处理增强')
# 锐化对比
sharp_0 = PostProcessor.sharpen(res_laplacian, 0)
sharp_1 = PostProcessor.sharpen(res_laplacian, 1.0)
sharp_2 = PostProcessor.sharpen(res_laplacian, 2.0)
sharp_compare = np.hstack([
    cv2.resize(sharp_0, (250, 200)), cv2.resize(sharp_1, (250, 200)), cv2.resize(sharp_2, (250, 200))
])
cv2.putText(sharp_compare, 'strength=0', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.putText(sharp_compare, 'strength=1.0', (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.putText(sharp_compare, 'strength=2.0', (510, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
save(sharp_compare, '14_post_sharpen.png')

# 对比度/亮度对比
contrast_low = PostProcessor.adjust_contrast_brightness(res_laplacian, 0.5, 0)
contrast_high = PostProcessor.adjust_contrast_brightness(res_laplacian, 1.5, 0)
bright_high = PostProcessor.adjust_contrast_brightness(res_laplacian, 1.0, 30)
contrast_compare = np.hstack([
    cv2.resize(contrast_low, (200, 160)), cv2.resize(res_laplacian, (200, 160)),
    cv2.resize(contrast_high, (200, 160)), cv2.resize(bright_high, (200, 160))
])
cv2.putText(contrast_compare, 'alpha=0.5', (5, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
cv2.putText(contrast_compare, 'alpha=1.0', (205, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
cv2.putText(contrast_compare, 'alpha=1.5', (405, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
cv2.putText(contrast_compare, 'beta=30', (605, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255,255,255), 1)
save(contrast_compare, '15_post_contrast.png')

# 饱和度对比
sat_low = PostProcessor.color_adjust(res_laplacian, sat_scale=0.3)
sat_high = PostProcessor.color_adjust(res_laplacian, sat_scale=1.8)
sat_compare = np.hstack([
    cv2.resize(sat_low, (250, 200)), cv2.resize(res_laplacian, (250, 200)), cv2.resize(sat_high, (250, 200))
])
cv2.putText(sat_compare, 'sat=0.3', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.putText(sat_compare, 'sat=1.0', (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.putText(sat_compare, 'sat=1.8', (510, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
save(sat_compare, '16_post_saturation.png')

# 暗角效果对比
vig_0 = PostProcessor.vignette(res_laplacian, 0)
vig_1 = PostProcessor.vignette(res_laplacian, 0.4)
vig_2 = PostProcessor.vignette(res_laplacian, 0.8)
vig_compare = np.hstack([
    cv2.resize(vig_0, (250, 200)), cv2.resize(vig_1, (250, 200)), cv2.resize(vig_2, (250, 200))
])
cv2.putText(vig_compare, 'strength=0', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.putText(vig_compare, 'strength=0.4', (260, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
cv2.putText(vig_compare, 'strength=0.8', (510, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 2)
save(vig_compare, '17_post_vignette.png')

# ── 6. 不同场景效果 ──
print('\n[6] 不同场景合成')
# 室内背景
bg_indoor = cv2.imread(r'D:\AiStdio\Python\PythonProject\DIP\dataset\background\indoor_room.jpg')
fg_s = Preprocessor.resize_to_match(
    Preprocessor.denoise(fg, 'bilateral'), bg_indoor)
m_s = cv2.resize(mask_refined, (fg_s.shape[1], fg_s.shape[0]))
m_s = m_s[..., np.newaxis] if m_s.ndim == 2 else m_s
res_scene1 = ImageCompositor.laplacian_pyramid_blend(fg_s, bg_indoor, m_s, pos)
save(res_scene1, '18_scene_indoor.png')

# 夜景背景
fg_s2 = Preprocessor.resize_to_match(
    Preprocessor.denoise(fg, 'bilateral'), bg2)
m_s2 = cv2.resize(mask_refined, (fg_s2.shape[1], fg_s2.shape[0]))
m_s2 = m_s2[..., np.newaxis] if m_s2.ndim == 2 else m_s2
res_scene2 = ImageCompositor.laplacian_pyramid_blend(fg_s2, bg2, m_s2, pos)
save(res_scene2, '19_scene_night.png')

# 花朵前景
fg2_denoised = Preprocessor.denoise(fg2, 'bilateral')
mask_f2_bin = ForegroundExtractor.color_threshold(fg2_denoised, 'white')
mask_f2 = ForegroundExtractor.refine_mask(mask_f2_bin)
fg2_s = Preprocessor.resize_to_match(fg2_denoised, bg)
m2_s = cv2.resize(mask_f2, (fg2_s.shape[1], fg2_s.shape[0]))
m2_s = m2_s[..., np.newaxis] if m2_s.ndim == 2 else m2_s
res_scene3 = ImageCompositor.laplacian_pyramid_blend(fg2_s, bg, m2_s, pos)
save(res_scene3, '20_scene_flower.png')

# ── 7. 完整处理流程总览 ──
print('\n[7] 完整处理流程总览')
# 读取已有步骤图重新排列
preview_list = [
    ('01_fg_original.png', '原图'),
    ('03b_fg_gray.png', '灰度化'),
    ('04_fg_denoised.png', '双边滤波去噪'),
    ('03_prep_fg_clahe.png', 'CLAHE增强'),
    ('05c_mask_grabcut_before_morph.png', 'GrabCut原始掩码'),
    ('05_mask_grabcut.png', '形态学后处理'),
    ('08_mask_refined.png', '边缘羽化'),
    ('10_result_laplacian.png', '最终合成'),
]
imgs_for_pipeline = []
for fname, label in preview_list:
    path = os.path.join(OUT, fname)
    if os.path.exists(path):
        img = cv2.imread(path)
        img_r = cv2.resize(img, (160, 120))
        cv2.putText(img_r, label, (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,0,0), 1)
        imgs_for_pipeline.append(img_r)

if len(imgs_for_pipeline) >= 8:
    row1 = np.hstack(imgs_for_pipeline[:4])
    row2 = np.hstack(imgs_for_pipeline[4:8])
    max_w = max(row1.shape[1], row2.shape[1])
    if row1.shape[1] < max_w:
        row1 = cv2.copyMakeBorder(row1, 0, 0, 0, max_w - row1.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255))
    if row2.shape[1] < max_w:
        row2 = cv2.copyMakeBorder(row2, 0, 0, 0, max_w - row2.shape[1], cv2.BORDER_CONSTANT, value=(255,255,255))
    pipeline = np.vstack([row1, row2])
    save(pipeline, '21_pipeline_overview.png')

print(f'\n所有图片已保存至: {OUT}')
print(f'共 {len(os.listdir(OUT))} 个文件')
