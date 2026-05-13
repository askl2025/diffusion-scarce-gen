# -*- coding: utf-8 -*-
import json
import os
import shutil
from collections import defaultdict

def calculate_scarcity_score(bbox, occlusion, truncation, category_name, category_ratios):
    """
    计算目标的稀缺性分数 (0-100)
    """
    # 1. 目标尺寸分数 (高度越小越难识别)
    height = bbox[3]
    if height < 25: size_score = 100
    elif height < 40: size_score = 60
    elif height < 80: size_score = 30
    else: size_score = 10

    # 2. 遮挡与截断 (CODA 特有属性)
    occlusion_score = occlusion * 40 
    truncation_score = truncation * 80

    # 3. 类别频率分数 (在稀缺类别中，占比越低的得分越高)
    ratio = category_ratios.get(category_name, 0.01)
    category_score = (1 - ratio) * 50

    total = size_score + occlusion_score + truncation_score + category_score
    return min(100, total)

def filter_rare_categories_and_samples(json_path, image_dir, output_root, cat_threshold=0.05, top_k=50):
    """
    1. 统计占比 -> 2. 筛选稀缺类别 (占比 < cat_threshold) -> 3. 评分 -> 4. 提取 Top K
    :param cat_threshold: 类别占比阈值。例如 0.05 表示只处理在数据集中占比低于 5% 的类别。
    """
    if not os.path.exists(output_root):
        os.makedirs(output_root)

    # 1. 加载数据
    print(f"正在读取 JSON: {json_path}")
    with open(json_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    cat_map = {cat['id']: cat['name'] for cat in data['categories']}
    img_map = {img['id']: img for img in data['images']}
    
    # 2. 统计全局分布并预筛选稀缺类别
    cat_counts = defaultdict(int)
    for ann in data['annotations']:
        cat_counts[cat_map[ann['category_id']]] += 1
    
    total_ann = sum(cat_counts.values())
    cat_ratios = {name: count / total_ann for name, count in cat_counts.items()}
    
    # 筛选出稀缺类别名单
    rare_category_names = [name for name, ratio in cat_ratios.items() if ratio < cat_threshold]
    
    print("\n--- 类别分布统计与筛选结果 ---")
    print(f"{'类别名称':<15} | {'占比':<8} | {'状态'}")
    print("-" * 40)
    for name, ratio in sorted(cat_ratios.items(), key=lambda x: x[1]):
        status = "★ 选定为稀缺类别" if name in rare_category_names else "数量充足(略过)"
        print(f"{name:<15} | {ratio:>7.2%} | {status}")

    # 3. 针对选定的稀缺类别进行评分
    scarcity_pools = defaultdict(list)
    print(f"\n正在对 {len(rare_category_names)} 个稀缺类别进行困难样本打分...")

    for ann in data['annotations']:
        cat_name = cat_map[ann['category_id']]
        
        # 只处理选中的稀缺类别
        if cat_name not in rare_category_names:
            continue
            
        occlusion = ann.get('attributes', {}).get('occlusion', 0)
        truncation = ann.get('attributes', {}).get('truncation', 0.0)
        
        score = calculate_scarcity_score(ann['bbox'], occlusion, truncation, cat_name, cat_ratios)

        sample_info = {
            "image_id": ann['image_id'],
            "file_name": img_map[ann['image_id']]['file_name'],
            "scarcity_score": round(score, 2),
            "category": cat_name,
            "bbox": ann['bbox']
        }
        scarcity_pools[cat_name].append(sample_info)

    # 4. 排序并复制图片
    final_log = {}
    print(f"\n--- 开始提取 Top-{top_k} 困难样本图片 ---")

    for cat_name in rare_category_names:
        samples = scarcity_pools[cat_name]
        # 按稀缺度分数降序排列
        samples.sort(key=lambda x: x['scarcity_score'], reverse=True)
        top_samples = samples[:top_k]
        final_log[cat_name] = top_samples

        if not top_samples: continue

        # 建立子文件夹
        cat_folder = os.path.join(output_root, cat_name.replace(" ", "_"))
        os.makedirs(cat_folder, exist_ok=True)

        copied_images = set()
        for s in top_samples:
            if s['file_name'] in copied_images: continue
            
            src = os.path.join(image_dir, s['file_name'])
            dst = os.path.join(cat_folder, s['file_name'])

            if os.path.exists(src):
                shutil.copy2(src, dst)
                copied_images.add(s['file_name'])
        
        print(f"已完成类别 [{cat_name}]: 提取了 {len(copied_images)} 张最高分图片")

    # 5. 保存 JSON 报告
    report_path = os.path.join(output_root, "rare_categories_report.json")
    with open(report_path, 'w', encoding='utf-8') as f:
        json.dump(final_log, f, indent=4, ensure_ascii=False)

    print(f"\n 处理完成！报告位置: {report_path}")

if __name__ == "__main__":
    # TODO: 配置你的数据路径
    JSON_PATH = r'path/to/your/annotations.json'    # CODA标注文件路径
    IMAGE_DIR = r'path/to/your/images'               # 图像目录
    OUTPUT_DIR = r'path/to/your/output'              # 输出目录

    # cat_threshold=0.3 表示：只处理数量占比低于 30% 的"小众"类别
    # top_k=50 表示：在这些小众类别里，选出前 50 个最难识别的样本
    filter_rare_categories_and_samples(JSON_PATH, IMAGE_DIR, OUTPUT_DIR, cat_threshold=0.3, top_k=50)