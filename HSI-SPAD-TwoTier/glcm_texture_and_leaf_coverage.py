import numpy as np
import matplotlib.pyplot as plt
from spectral import envi
import cv2
import tqdm
from pathlib import Path
import traceback
from skimage import morphology
from skimage.feature import graycomatrix
from sklearn.decomposition import PCA
import pandas as pd


# ==========================================
# 1. 核心特征计算函数
# ==========================================
def haralick_features(P_raw):
    # 消除背景干扰！
    P = P_raw.copy().astype(np.float64)
    P[0, :] = 0  # 消除背景行
    P[:, 0] = 0  # 消除背景列

    eps = 1e-10
    total_sum = P.sum()
    if total_sum == 0:
        return {k: 0 for k in ["ASM", "contrast", "correlation", "variance", "homogeneity",
                               "entropy", "sum_avg", "sum_var", "sum_entropy",
                               "diff_var", "diff_entropy", "IMC1", "IMC2", "MCC"]}

    P = P / total_sum  # 手动重新归一化
    Ng = P.shape[0]

    i, j = np.indices((Ng, Ng))
    px = np.sum(P, axis=1)
    py = np.sum(P, axis=0)

    ux = np.sum(i * P)
    uy = np.sum(j * P)

    sigx = np.sqrt(np.sum((i - ux) ** 2 * P))
    sigy = np.sqrt(np.sum((j - uy) ** 2 * P))

    # 基础特征
    ASM = np.sum(P ** 2)
    contrast = np.sum((i - j) ** 2 * P)
    correlation = np.sum((i - ux) * (j - uy) * P) / (sigx * sigy + eps)
    variance = np.sum((i - ux) ** 2 * P)
    homogeneity = np.sum(P / (1 + (i - j) ** 2))
    entropy = -np.sum(P * np.log(P + eps))

    # Sum 特征
    p_sum = np.array([np.sum(P[i + j == k]) for k in range(2 * Ng)])
    sum_avg = np.sum(np.arange(len(p_sum)) * p_sum)
    sum_var = np.sum((np.arange(len(p_sum)) - sum_avg) ** 2 * p_sum)
    sum_entropy = -np.sum(p_sum * np.log(p_sum + eps))

    # Diff 特征 (修正的公式)
    p_diff = np.array([np.sum(P[np.abs(i - j) == k]) for k in range(Ng)])
    diff_avg = np.sum(np.arange(Ng) * p_diff)
    diff_var = np.sum((np.arange(Ng) - diff_avg) ** 2 * p_diff)
    diff_entropy = -np.sum(p_diff * np.log(p_diff + eps))

    # IMC 特征
    HX = -np.sum(px * np.log(px + eps))
    HY = -np.sum(py * np.log(py + eps))
    HXY = entropy
    HXY1 = -np.sum(P * np.log(px[i] * py[j] + eps))
    IMC1 = (HXY - HXY1) / (max(HX, HY) + eps)
    HXY2 = -np.sum(px[i] * py[j] * np.log(px[i] * py[j] + eps))
    IMC2 = np.sqrt(1 - np.exp(-2 * (HXY2 - HXY)))

    # MCC 特征
    px_safe = px + eps
    py_safe = py + eps
    Q = np.zeros((Ng, Ng))
    for i_idx in range(Ng):
        for j_idx in range(Ng):
            Q[i_idx, j_idx] = np.sum((P[i_idx, :] * P[j_idx, :]) / (px_safe[i_idx] * py_safe))
    try:
        eigenvalues = np.sort(np.abs(np.linalg.eigvals(Q)))
        second_largest_eigenvalue = eigenvalues[-2] if len(eigenvalues) >= 2 else 0
        MCC = np.sqrt(second_largest_eigenvalue)
    except Exception:
        MCC = 0.0

    return {
        "ASM": ASM, "contrast": contrast, "correlation": correlation, "variance": variance,
        "homogeneity": homogeneity, "entropy": entropy, "sum_avg": sum_avg,
        "sum_var": sum_var, "sum_entropy": sum_entropy, "diff_var": diff_var,
        "diff_entropy": diff_entropy, "IMC1": IMC1, "IMC2": IMC2, "MCC": MCC
    }


# ==========================================
# 2. 单个文件的处理主流程
# ==========================================
def process_single_image(hdr_path, spe_path, save_dir):
    file_name = hdr_path.stem  # 获取文件名，不带后缀 (例如 'HS-99')
    print(f"\n---> 开始处理: {file_name}")

    # 1. 读取数据
    img = envi.open(str(hdr_path), str(spe_path))
    data = np.array(img.load()).astype(np.float32)
    h, w, bands = data.shape
    wavelengths = np.array(img.metadata['wavelength'], dtype=float)

    # 2. 白板校正
    band_130_idx = 129
    white_ref = data[:, :, band_130_idx]
    threshold = np.percentile(white_ref, 98)
    white_mask = white_ref > threshold
    white_mean = np.zeros(bands)
    for b in range(bands):
        vals = data[:, :, b][white_mask]
        white_mean[b] = np.mean(vals) if len(vals) > 0 else np.mean(data[:, :, b])

    calibrated = np.zeros_like(data)
    for b in range(bands):
        if white_mean[b] > 1e-6:
            calibrated[:, :, b] = (data[:, :, b] / white_mean[b]) * 255
    data = calibrated

    # 3. 玉米 Mask
    maize_mask = (data[:, :, 75] - data[:, :, 51]) >= 5
    soil_mask = (data[:, :, 163] - data[:, :, 153]) >= 10
    white_mask = data[:, :, band_130_idx] > 250
    mask = maize_mask & soil_mask & ~white_mask
    mask = morphology.remove_small_objects(mask, min_size=100)

    total_pixels = h * w  # 图片总像素 (例如 480*480)
    leaf_pixels_n = np.sum(mask)  # 叶片总像素 (Mask中为 True 的数量)
    leaf_ratio = leaf_pixels_n / total_pixels  # 占比比例


    # 4. 光谱提取
    rows, cols = np.where(mask)
    N = len(rows)
    spectra = np.zeros((N, bands))
    # 为避免批量跑时满屏进度条，这里去掉了 tqdm
    for i in range(N):
        spectra[i] = data[rows[i], cols[i], :]

    np.savez(save_dir / f"{file_name}_spectra.npz",
             spectra=spectra, coords=np.column_stack((rows, cols)), wavelengths=wavelengths)

    # ================= 5. 提取指定波段生成 RGB=================
    # 5. 提取指定波段生成 RGB，并转为灰度图提取纹理
    r_band = data[:, :, 129]
    g_band = data[:, :, 79]
    b_band = data[:, :, 29]

    # 合成RGB
    rgb = np.stack([r_band, g_band, b_band], axis=2)
    rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    rgb[~mask] = 0  # 背景置0

    # 将 RGB 转为单通道灰度图，以满足 GLCM 输入要求
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    roi = gray

    # =================  6. GLCM 步骤 =================
    # 6. GLCM
    levels = 64
    roi_q = np.zeros_like(roi, dtype=np.uint8)

    # 将 mask 区域内的灰度值线性映射到 1 ~ levels (即 1~64) 之间，背景保留为 0
    # 使用统一映射（所有图像相同）：0-255 → 1-64
    # 这样可以确保不同图像之间的纹理特征可以直接比较
    roi_q[mask] = (roi[mask].astype(float) / 4 + 1).astype(np.uint8)  # 256/64=4
    roi_q[mask] = np.clip(roi_q[mask], 1, 64)

    glcm = graycomatrix(roi_q, distances=[1], angles=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
                        levels=levels, symmetric=True, normed=False)

    # 7. Haralick 14特征提取
    features_list = []
    for d in range(glcm.shape[2]):
        for a in range(glcm.shape[3]):
            P = glcm[:, :, d, a]
            features_list.append(haralick_features(P))

    df = pd.DataFrame(features_list)
    final_features = df.mean().to_dict()

    # 在特征字典最前面插入文件名，方便识别
    result_dict = {"File_Name": file_name}
    result_dict.update(final_features)
    result_dict.update(final_features)
    result_dict2 = {
        "File_Name": file_name,
        "Leaf_Pixels": leaf_pixels_n,
        "Leaf_Ratio": leaf_ratio
    }


    # ================= 保存灰度图和RGB图 =================

    plt.imsave(save_dir / f"{file_name}_gray.png", gray, cmap='gray')
    # 彩色的RGB

    plt.imsave(save_dir / f"{file_name}_rgb.png", rgb)

    return result_dict,result_dict2


# ==========================================
# 3. 批量运行主程序
# ==========================================
if __name__ == "__main__":
    # 配置路径
    base_dir = Path(__file__).parent
    save_dir = base_dir / 'output' / 'texture_analysis'
    save_dir.mkdir(parents=True, exist_ok=True)

    # 定义两个数据目录
    data_dirs = {
        'CK': Path(r'C:\Users\13004\Desktop\SPAD\shude\CK'),
        'DS': Path(r'C:\Users\13004\Desktop\SPAD\shude\实验')
    }

    # 处理每个数据集
    for group_name, data_dir in data_dirs.items():
        print(f"\n{'='*60}")
        print(f"开始处理 {group_name} 组数据")
        print(f"{'='*60}")

        # 获取所有 .hdr 文件
        hdr_files = list(data_dir.glob("*.hdr"))

        if not hdr_files:
            print(f"在 {data_dir} 下没有找到任何 .hdr 文件，请检查路径。")
            continue
        else:
            print(f"共找到 {len(hdr_files)} 个数据文件，准备开始批量处理...")

        all_results = []
        all_pixels = []
        # 遍历所有文件
        for hdr_path in tqdm.tqdm(hdr_files, desc=f"{group_name}组进度"):
            # 推导对应的 .spe 文件路径
            spe_path = hdr_path.with_suffix('.spe')

            if not spe_path.exists():
                print(f"\n警告: 找不到 {hdr_path.name} 对应的 .spe 文件，已跳过。")
                continue

            try:
                # 调用核心处理函数
                res, pixels = process_single_image(hdr_path, spe_path, save_dir)
                all_results.append(res)
                all_pixels.append(pixels)
            except Exception as e:
                print(f"\n处理 {hdr_path.name} 时发生错误: {e}")
                traceback.print_exc()
                continue  # 发生错误时跳过该文件，继续下一个

        # 保存表格
        if all_results:
            final_df_results = pd.DataFrame(all_results)
            output_xlsx_1 = save_dir / f"{group_name}_haralick_features.xlsx"
            final_df_results.to_excel(output_xlsx_1, index=False)

            print("\n" + "=" * 40)
            print(f"{group_name}组特征批量处理完成！共成功处理 {len(all_results)} 个样本。")
            print(f"总特征数据表已保存至: {output_xlsx_1}")
            print("=" * 40)

        if all_pixels:
            final_df_pixels = pd.DataFrame(all_pixels)
            output_xlsx_2 = save_dir / f"{group_name}_pixels_info.xlsx"
            final_df_pixels.to_excel(output_xlsx_2, index=False)

            print("\n" + "=" * 40)
            print(f"{group_name}组像素批量处理完成！共成功处理 {len(all_pixels)} 个样本。")
            print(f"总像素数据表已保存至: {output_xlsx_2}")
            print("=" * 40)

    if all_pixels:
        final_df_pixels = pd.DataFrame(all_pixels)
        output_xlsx_2 = save_dir / "CK_pixels_info.xlsx"
        final_df_pixels.to_excel(output_xlsx_2, index=False)

        print("\n" + "=" * 40)
        print(f"像素批量处理完成！共成功处理 {len(all_pixels)} 个样本。")
        print(f"总像素数据表已保存至: {output_xlsx_2}")
        print("=" * 40)