import os
import argparse
import random

def count_images_in_folder(folder_path):
    """计算指定文件夹中图片数量"""
    if not os.path.exists(folder_path):
        return 0
    try:
        # 兼容常见图片格式
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        return len(image_files)
    except Exception as e:
        return 0

def process_list_file(list_file_path, image_root_dir):
    """
    读取官方的 list txt 文件，解析路径和标签，并检查本地图片文件夹
    """
    valid_data = []
    missing_count = 0
    
    print(f"正在读取列表文件: {list_file_path} ...")
    
    with open(list_file_path, 'r') as f:
        lines = f.readlines()

    total_lines = len(lines)
    print(f"列表文件中共有 {total_lines} 条记录。开始匹配本地图片文件夹...")

    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        parts = line.split()
        if len(parts) < 2:
            print(f"跳过格式错误行: {line}")
            continue
            
        # 解析：你的文件格式是 "1 Path/To/Video.mp4"
        # parts[0] 是标签, parts[1] 是视频相对路径
        raw_label = int(parts[0])
        video_rel_path = parts[1]

        # =======================================================
        # 标签转换逻辑 (根据用户要求)
        # 输入文件: 1 = 真 (Real), 0 = 假 (Fake)
        # 输出文件: 0 = 真 (Real), 1 = 假 (Fake)
        # =======================================================
        final_label = 1 - raw_label
        
        # 1. 处理路径：去掉 .mp4 后缀，因为图片文件夹通常没有后缀
        # 例如: Celeb-real/id57_0003.mp4 -> Celeb-real/id57_0003
        folder_rel_path = os.path.splitext(video_rel_path)[0]
        
        # 2. 拼接完整的图片文件夹路径
        # 例如: /data/.../Celeb-DF-v3/ + Celeb-real/id57_0003
        full_folder_path = os.path.join(image_root_dir, folder_rel_path)
        
        # 3. 检查文件夹是否存在并计算图片数量
        if os.path.exists(full_folder_path):
            num_images = count_images_in_folder(full_folder_path)
            if num_images > 0:
                # 找到了！添加到列表
                # 格式: (完整路径, 起始帧, 结束帧, 转换后的标签)
                valid_data.append((full_folder_path, 0, num_images - 1, final_label))
            else:
                # 文件夹存在但没图片
                print(f"[警告] 文件夹为空: {full_folder_path}")
                missing_count += 1
        else:
            # 文件夹根本找不到（可能是路径不对，或者你没提取这个视频的帧）
            # 调试打印，只打印前5个找不到的，避免刷屏
            if missing_count < 5:
                print(f"[提示] 未找到对应文件夹 (可能路径不匹配?): {full_folder_path}")
            missing_count += 1

    print(f"\n--- 扫描统计 ---")
    print(f"列表总数: {total_lines}")
    print(f"成功匹配: {len(valid_data)}")
    print(f"缺失/无效: {missing_count}")
    
    if missing_count > 0:
        print(f"⚠️ 注意: 有 {missing_count} 个视频没找到对应的图片文件夹。请检查 --image_root 是否包含了 Celeb-real 和 Celeb-synthesis 的上一级目录。")
        
    return valid_data

def write_output(filepath, data):
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        for path, start, end, label in data:
            # 格式: 路径 起始帧 结束帧 标签 (空格分隔)
            f.write(f"{path} {start} {end} {label}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="基于 List_of_testing_videos.txt 生成测试集配置")

    # 1. 官方列表文件的路径
    parser.add_argument("--list_file", type=str, default='/data/siyu.liu/datasets/Celeb-DF-v3/List_of_testing_videos.txt',
                        help="包含 '1 Path/To/Video.mp4' 格式的 txt 文件路径")
    
    # 2. 图片的根目录
    # 注意：这个目录下应该直接包含 'Celeb-real', 'Celeb-synthesis' 等文件夹
    parser.add_argument("--image_root", type=str, default='/data/siyu.liu/datasets/Celeb-DF-v3/Celeb_image/',
                        help="图片数据的根目录 (包含 Celeb-real/Celeb-synthesis 子文件夹的地方)")
    
    # 3. 输出目录
    parser.add_argument("--output_dir", type=str, default='/data/siyu.liu/datasets/Celeb-DF-v3/',
                        help="生成的 test.txt 存放目录")
    
    # 4. (可选) 是否要从这里面分出验证集
    parser.add_argument("--val_ratio", type=float, default=0.0,
                        help="从列表中切分出验证集的比例 (0.0~1.0)。默认 0.0 (全部作为测试集)")

    args = parser.parse_args()

    # --- 执行处理 ---
    all_data = process_list_file(args.list_file, args.image_root)

    if not all_data:
        print("❌ 错误: 没有匹配到任何有效数据，程序退出。")
        exit()

    # --- 随机打乱 ---
    random.shuffle(all_data)

    # --- 切分逻辑 ---
    total_count = len(all_data)
    val_count = int(total_count * args.val_ratio)
    test_count = total_count - val_count

    val_data = all_data[:val_count]
    test_data = all_data[val_count:]

    # --- 写入文件 ---
    print(f"\n正在写入文件...")
    
    # 写入测试集
    test_txt_path = os.path.join(args.output_dir, "test.txt")
    write_output(test_txt_path, test_data)
    print(f"✅ 生成 {test_txt_path} (共 {len(test_data)} 条)")

    # 如果有验证集，写入验证集
    if val_count > 0:
        val_txt_path = os.path.join(args.output_dir, "val.txt")
        write_output(val_txt_path, val_data)
        print(f"✅ 生成 {val_txt_path} (共 {len(val_data)} 条)")

    print("\n完成！")