import json
import os
from tqdm import tqdm

def convert_coda_to_kitti(json_path, output_dir):
    """
    解析 CODA 原生 COCO 格式并无损映射至 KITTI 格式
    :param json_path: CODA 标注文件路径 (如 annotations.json)
    :param output_dir: KITTI 格式标签存储目录
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 1. 加载 CODA 标注数据
    print(f"正在加载标注文件: {json_path}...")
    with open(json_path, 'r', encoding='utf-8') as f:
        coco_data = json.load(f)

    # 2. 构建 ID 到名称/文件名的映射映射
    # 获取类别映射 (ID -> Name)
    cat_id_to_name = {cat['id']: cat['name'] for cat in coco_data['categories']}
    
    # 获取图像映射 (ID -> FileName)
    # KITTI 标签文件名通常与图像名对应（不含扩展名）
    img_id_to_info = {img['id']: os.path.splitext(img['file_name'])[0] for img in coco_data['images']}

    # 3. 按图像 ID 归档标注
    ann_by_img = {}
    for ann in coco_data['annotations']:
        img_id = ann['image_id']
        if img_id not in ann_by_img:
            ann_by_img[img_id] = []
        ann_by_img[img_id].append(ann)

    # 4. 执行转换与坐标对齐
    print("正在转换格式并对齐坐标系...")
    for img_id, img_name in tqdm(img_id_to_info.items()):
        label_file_path = os.path.join(output_dir, f"{img_name}.txt")
        anns = ann_by_img.get(img_id, [])

        with open(label_file_path, 'w', encoding='utf-8') as f:
            for ann in anns:
                # 获取类别名
                obj_type = cat_id_to_name[ann['category_id']]
                
                # 坐标系对齐: [x_min, y_min, w, h] -> [left, top, right, bottom]
                bbox = ann['bbox']
                left = float(bbox[0])
                top = float(bbox[1])
                right = left + float(bbox[2])
                bottom = top + float(bbox[3])

                # 构建 KITTI 15 列标准格式
                # 1. Type: 类别名
                # 2. Truncated: 截断度 (默认 0.0)
                # 3. Occluded: 遮挡 (默认 0)
                # 4. Alpha: 观测角 (默认 -10)
                # 5-8. BBox: 对齐后的坐标
                # 9-11. Dimensions: 3D 尺寸 (高度, 宽度, 长度, 默认 -1)
                # 12-14. Location: 3D 中心坐标 (x, y, z, 默认 -1)
                # 15. Rotation_y: 偏航角 (默认 -10)
                
                kitti_line = (
                    f"{obj_type} 0.0 0 -10 "
                    f"{left:.2f} {top:.2f} {right:.2f} {bottom:.2f} "
                    f"-1 -1 -1 -1 -1 -1 -10\n"
                )
                f.write(kitti_line)

    print(f"转换完成！KITTI 标签已保存至: {output_dir}")

if __name__ == "__main__":
    # 配置路径
    CODA_ANNOTATIONS = "annotations.json"  # CODA 标注文件路径
    OUTPUT_PATH = "kitti_labels"           # 输出目录
    
    convert_coda_to_kitti(CODA_ANNOTATIONS, OUTPUT_PATH)