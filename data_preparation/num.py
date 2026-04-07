import os

def read_input_file(input_file):
    """
    Reads the input txt file and returns a list of tuples containing
    (folder_path, start_frame, end_frame, label)
    """
    with open(input_file, 'r') as file:
        lines = file.readlines()
    data = [line.strip().split() for line in lines]
    return [(d[0], int(d[1]), int(d[2]), d[3]) for d in data]

def count_images_in_folder(folder_path):
    """
    Counts the number of image files in the specified folder.
    """
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(('.jpg'))]
    return len(image_files)

def update_end_frames(data):
    """
    Updates the end frames for each entry in the data list based on the number of images in each folder.
    """
    updated_data = []
    for folder_path, start_frame, end_frame, label in data:
        num_images = count_images_in_folder(folder_path)
        new_end_frame = num_images - 1  # 更新结束帧为图像总数减1
        updated_data.append((folder_path, start_frame, new_end_frame, label))
    return updated_data

def write_output_file(output_file, updated_data):
    """
    Writes the updated data to the specified output txt file.
    """
    with open(output_file, 'w') as file:
        for folder_path, start_frame, end_frame, label in updated_data:
            file.write(f"{folder_path} {start_frame} {end_frame} {label}\n")

if __name__ == "__main__":
    input_file = "/data/siyu.liu/datasets/FF++_c40/val.txt"  # 替换为你的输入txt文件路径
    output_file = "/data/siyu.liu/datasets/FF++_c40/re-val.txt"  # 替换为你的输出txt文件路径

    # 读取输入文件
    data = read_input_file(input_file)
    
    # 更新结束帧
    updated_data = update_end_frames(data)
    
    # 写入输出文件
    write_output_file(output_file, updated_data)

    print("Updated end frames and wrote to output file.")
