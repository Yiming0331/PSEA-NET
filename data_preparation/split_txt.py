# import os
# import random
# import argparse # 导入argparse模块，用于解析命令行参数

# def count_images_in_folder(folder_path):
#     """
#     计算指定文件夹中 .jpg 图像文件的数量。
#     (此函数来自 num.py)
#     """
    
#     # 检查路径是否存在，如果不存在则返回0
#     if not os.path.exists(folder_path):
#         print(f"警告：文件夹不存在, 路径: {folder_path}")
#         return 0
        
#     try:
#         # 尝试列出文件
#         image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg'))]
#         return len(image_files)
#     except Exception as e:
#         # 处理可能的权限问题或其他错误
#         print(f"错误：无法读取文件夹 {folder_path}。错误信息: {e}")
#         return 0


# def write_output_file(output_file, updated_data):
#     """
#     将更新后的数据写入指定的输出 txt 文件。
#     (基于 num.py 修改)
    
#     关键修改：使用 'a' 模式（追加）而不是 'w' 模式（覆盖）。
#     """
#     # 使用 'a' (append) 模式打开文件，如果文件不存在则会创建它
#     with open(output_file, 'a') as file:
#         for folder_path, start_frame, end_frame, label in updated_data:
#             # 写入格式化的行，并添加换行符
#             file.write(f"{folder_path} {start_frame} {end_frame} {label}\n")

# def process_and_split_folders(input_folder, output_dir, label):
#     """
#     主处理函数：
#     1. 扫描输入文件夹中的所有子文件夹。
#     2. 随机打乱它们。
#     3. 按 70/15/15 比例切分。
#     4. 为每个子文件夹计算帧数。
#     5. 将结果追加写入 train.txt, val.txt, test.txt。
#     """
    
#     print(f"--- 开始处理 ---")
#     print(f"输入文件夹: {input_folder}")
#     print(f"统一标签: {label}")

#     # 1. 扫描所有子文件夹
#     try:
#         all_subfolders = [os.path.join(input_folder, f) for f in os.listdir(input_folder) 
#                           if os.path.isdir(os.path.join(input_folder, f))]
#     except Exception as e:
#         print(f"错误：无法读取输入文件夹 {input_folder}。错误信息: {e}")
#         return

#     if not all_subfolders:
#         print(f"警告：在 {input_folder} 中没有找到任何子文件夹。")
#         return

#     print(f"总共找到 {len(all_subfolders)} 个子文件夹。")

#     # 2. 随机打乱列表
#     random.shuffle(all_subfolders)

#     # 3. 计算切分点
#     total_count = len(all_subfolders)
#     train_count = int(total_count * 0.70)
#     val_count = int(total_count * 0.15)
#     # 剩下的是测试集，这样可以确保所有数据都被分配，即使有舍入误差
    
#     # 4. 执行切分
#     train_folders = all_subfolders[:train_count]
#     val_folders = all_subfolders[train_count : train_count + val_count]
#     test_folders = all_subfolders[train_count + val_count :]

#     print(f"切分结果: 训练集={len(train_folders)}, 验证集={len(val_folders)}, 测试集={len(test_folders)}")

#     # 5. 处理每个列表并准备写入数据
#     data_lists = {
#         "train": [],
#         "val": [],
#         "test": []
#     }

#     # 辅助函数，用于处理一个文件夹列表
#     def process_list(folder_list, list_name):
#         print(f"正在处理 {list_name} 列表...")
#         processed_data = []
#         for folder_path in folder_list:
#             # 调用 num.py 中的函数来计算图片数量
#             num_images = count_images_in_folder(folder_path)
            
#             if num_images == 0:
#                 print(f"警告：在 {folder_path} 中未找到 .jpg 图片，已跳过。")
#                 continue
                
#             start_frame = 0  # 默认开始帧为 0
#             end_frame = num_images - 1  # 结束帧为 (数量 - 1)，因为是0-based索引
            
#             # 将 (路径, 开始帧, 结束帧, 标签) 元组添加到列表中
#             processed_data.append((folder_path, start_frame, end_frame, label))
#         return processed_data

#     train_data = process_list(train_folders, "训练集")
#     val_data = process_list(val_folders, "验证集")
#     test_data = process_list(test_folders, "测试集")

#     # 6. 确保输出目录存在
#     os.makedirs(output_dir, exist_ok=True)

#     # 7. 将结果追加写入文件
#     train_file_path = os.path.join(output_dir, "train.txt")
#     val_file_path = os.path.join(output_dir, "val.txt")
#     test_file_path = os.path.join(output_dir, "test.txt")

#     write_output_file(train_file_path, train_data)
#     print(f"已追加 {len(train_data)} 条记录到 {train_file_path}")
    
#     write_output_file(val_file_path, val_data)
#     print(f"已追加 {len(val_data)} 条记录到 {val_file_path}")

#     write_output_file(test_file_path, test_data)
#     print(f"已追加 {len(test_data)} 条记录到 {test_file_path}")
    
#     print(f"--- 处理完成 ---")


# if __name__ == "__main__":
#     parser = argparse.ArgumentParser(description="处理图像文件夹并生成列表")
    
#     # ==========================================
#     #  请在这里直接修改 default='你的路径'
#     # ==========================================

#     # 输入文件夹路径 (例如 image_real 或 image_fake)
#     parser.add_argument("-i", "--input_folder", type=str, 
#                         default='/data/siyu.liu/datasets/SDFVD/image_fake_MTCNN/',  # <--- 在这里修改默认输入路径
#                         help="包含所有子图像文件夹的根目录")
    
#     # 输出 txt 的文件夹路径
#     parser.add_argument("-o", "--output_dir", type=str, 
#                         default='/data/siyu.liu/datasets/SDFVD/',       # <--- 在这里修改默认输出路径
#                         help="用于存放生成的 train.txt 等文件的目录")
    
#     # 标签 (0 或 1)
#     parser.add_argument("-l", "--label", type=int, choices=[0, 1], 
#                         default=1,                                           # <--- 在这里修改默认标签
#                         help="统一标签 (0=真, 1=假)")


#     # 3. 解析参数
#     args = parser.parse_args()

#     # 打印一下当前使用的参数，防止你忘了改
#     print(f"当前配置 -> 输入: {args.input_folder}")
#     print(f"当前配置 -> 输出: {args.output_dir}")
#     print(f"当前配置 -> 标签: {args.label}")


#     # 4. 调用主函数
#     process_and_split_folders(args.input_folder, args.output_dir, args.label)
import os
import random
import argparse

def count_images_in_folder(folder_path):
    """计算指定文件夹中 .jpg 图像文件的数量"""
    if not os.path.exists(folder_path):
        return 0
    try:
        image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg'))]
        return len(image_files)
    except Exception as e:
        print(f"错误读取: {e}")
        return 0

def write_output_file(output_file, updated_data):
    """将数据追加写入文件"""
    with open(output_file, 'a') as file:
        for folder_path, start_frame, end_frame, label in updated_data:
            file.write(f"{folder_path} {start_frame} {end_frame} {label}\n")

def process_and_split_folders(input_folder, output_dir, label):
    print(f"--- 开始处理 ---")
    print(f"输入根目录: {input_folder}")
    
    # ============================================================
    # 修改重点：使用 os.walk 递归寻找包含图片的底层文件夹
    # ============================================================
    all_subfolders = []
    print("正在递归扫描所有子目录，请稍候...")
    
    for root, dirs, files in os.walk(input_folder):
        # 检查当前文件夹(root)里有没有 .jpg 文件
        has_images = any(f.lower().endswith('.jpg') for f in files)
        
        if has_images:
            # 如果这个文件夹里有图片，它就是我们要的一个“数据单元”
            all_subfolders.append(root)

    if not all_subfolders:
        print(f"错误：在 {input_folder} 及其子目录中没有找到任何包含 .jpg 的文件夹。")
        return

    print(f"扫描完成！总共找到 {len(all_subfolders)} 个包含图片的文件夹。")
    # ============================================================

    # 2. 随机打乱列表 (混合来自三个大文件夹的所有数据)
    random.shuffle(all_subfolders)

    # 3. 计算切分点
    total_count = len(all_subfolders)
    train_count = int(total_count * 0.90)
    val_count = int(total_count * 0.05)
    
    # 4. 执行切分
    train_folders = all_subfolders[:train_count]
    val_folders = all_subfolders[train_count : train_count + val_count]
    test_folders = all_subfolders[train_count + val_count :]

    print(f"切分结果: 训练集={len(train_folders)}, 验证集={len(val_folders)}, 测试集={len(test_folders)}")

    # 5. 处理每个列表
    def process_list(folder_list, list_name):
        # 只有列表不为空时才打印进度，避免刷屏
        if folder_list:
            print(f"正在处理 {list_name} (共 {len(folder_list)} 个)...")
        
        processed_data = []
        for folder_path in folder_list:
            num_images = count_images_in_folder(folder_path)
            if num_images == 0:
                continue
            
            start_frame = 0
            end_frame = num_images - 1
            processed_data.append((folder_path, start_frame, end_frame, label))
        return processed_data

    train_data = process_list(train_folders, "训练集")
    val_data = process_list(val_folders, "验证集")
    test_data = process_list(test_folders, "测试集")

    # 6. 写入文件
    os.makedirs(output_dir, exist_ok=True)
    
    write_output_file(os.path.join(output_dir, "train_big_0.9.txt"), train_data)
    write_output_file(os.path.join(output_dir, "val_0.05.txt"), val_data)
    write_output_file(os.path.join(output_dir, "test_0.05.txt"), test_data)

    print(f"--- 处理完成: 已写入 {output_dir} ---")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # 输入文件夹路径 (例如 image_real 或 image_fake)
    parser.add_argument("-i", "--input_folder", type=str, 
                        default='/data/siyu.liu/datasets/Celeb-DF-v3/Celeb_image/Celeb-synthesis/TalkingFace/', 
                        help="包含所有子图像文件夹(需要是一个类的比如说都是“真的” 或者 都是“假的”)的根目录")
    
    # 输出 txt 的文件夹路径
    parser.add_argument("-o", "--output_dir", type=str, 
                        default='/data/siyu.liu/datasets/Celeb-DF-v3/', 
                        help="用于存放生成的 train.txt 等文件的目录")

    parser.add_argument("-l", "--label", type=int, default=1, help="统一标签 (0=真, 1=假)")
    
    args = parser.parse_args()

    # 打印一下当前使用的参数，防止你忘了改
    print(f"当前配置 -> 输入: {args.input_folder}")
    print(f"当前配置 -> 输出: {args.output_dir}")
    print(f"当前配置 -> 标签: {args.label}")

    process_and_split_folders(args.input_folder, args.output_dir, args.label)