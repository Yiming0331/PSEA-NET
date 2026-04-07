import os
import random

# ================= 您的配置区域 =================
base_dir = "/data/siyu.liu/datasets/FF++/3train_cross_one_manispa/"
file_prefixes = ["train", "test", "val"]

# 🔴 在这里修改你要剔除的方法 (Real, Face2Face, Deepfakes, FaceSwap, NeuralTextures)
target_to_exclude = "NeuralTextures" 
# ===========================================

def process_files():
    # 设置随机种子
    random.seed(42)

    for prefix in file_prefixes:
        input_filename = f"{prefix}.txt"
        
        # 自动根据配置生成文件名，例如: train_unNeuralTextures.txt
        output_filename = f"{prefix}_un{target_to_exclude}.txt"
        
        input_path = os.path.join(base_dir, input_filename)
        output_path = os.path.join(base_dir, output_filename)

        if not os.path.exists(input_path):
            print(f"⚠️ 跳过：找不到文件 {input_path}")
            continue

        print(f"正在处理: {input_filename} ...")
        print(f"👉 目标剔除: {target_to_exclude}")

        with open(input_path, 'r') as f:
            lines = f.readlines()

        original_count = len(lines)
        
        # --- 关键修改开始 ---
        processed_lines = []
        for line in lines:
            # 1. 过滤逻辑：使用变量 target_to_exclude
            if target_to_exclude in line:
                continue
            
            # 2. 清洗逻辑：去掉首尾空格和旧的换行符
            clean_line = line.strip()
            
            # 3. 只有非空行才保留，并且手动加上 "\n"
            if clean_line:
                processed_lines.append(clean_line + "\n")
        # --- 关键修改结束 ---

        #随机打乱
        random.shuffle(processed_lines)

        # 写入新文件
        with open(output_path, 'w') as f:
            f.writelines(processed_lines)

        print(f"✅ 完成！生成文件: {output_filename}")
        print(f"   - 原始行数: {original_count}")
        print(f"   - 过滤后行数: {len(processed_lines)}")
        print("-" * 30)

if __name__ == "__main__":
    process_files()