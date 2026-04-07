import random

# 1. 设置文件路径
input_file = "/data/siyu.liu/datasets/Celeb-DF-v3/test.txt"  # 你的原始文件路径
output_file = "/data/siyu.liu/datasets/Celeb-DF-v3/test_shuff.txt" # 你想保存的新路径

# 2. 读取所有行
with open(input_file, 'r') as f:
    lines = f.readlines()

# 3. 打乱顺序
# 设定种子，保证每次打乱的结果都一样（如果你不需要复现，可以把这行删掉）
random.seed(42) 
random.shuffle(lines)

# 4. 写入新文件
with open(output_file, 'w') as f:
    f.writelines(lines)

print(f"完成！已生成乱序文件: {output_file}，共 {len(lines)} 行。")