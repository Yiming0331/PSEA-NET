import os
import shutil

def rename_and_copy_images_in_folder(src_folder, dest_folder):
    """
    Rename images in the specified source folder to a sequential format starting from 000,
    and copy them to the destination folder.

    Args:
        src_folder (str): Path to the source folder containing images.
        dest_folder (str): Path to the destination folder to save renamed images.
    """
    if not os.path.exists(dest_folder):
        os.makedirs(dest_folder)
    
    # 获取文件夹中的所有文件并按名称排序
    files = sorted(os.listdir(src_folder))

    # 过滤出图像文件（可以根据需要添加更多的扩展名）
    image_files = [f for f in files if f.lower().endswith(('.jpg'))]

    # 遍历图像文件并重新命名
    for i, filename in enumerate(image_files):
        # 生成新的文件名，例如 000.jpg
        new_name = f"{i:03d}{os.path.splitext(filename)[1]}"
        
        # 获取旧文件名和新文件名的完整路径
        old_file = os.path.join(src_folder, filename)
        new_file = os.path.join(dest_folder, new_name)
        
        # 复制并重命名文件
        shutil.copy(old_file, new_file)
        print(f"Copied and renamed {old_file} to {new_file}")

def process_all_subfolders(src_root_folder, dest_root_folder):
    """
    Process all subfolders in the specified source root folder and copy renamed images to the destination root folder.

    Args:
        src_root_folder (str): Path to the source root folder containing image folders.
        dest_root_folder (str): Path to the destination root folder to save renamed image folders.
    """
    for subdir, dirs, _ in os.walk(src_root_folder):
        for dir_name in dirs:
            src_folder = os.path.join(subdir, dir_name)
            relative_path = os.path.relpath(src_folder, src_root_folder)
            dest_folder = os.path.join(dest_root_folder, relative_path)
            rename_and_copy_images_in_folder(src_folder, dest_folder)

if __name__ == "__main__":
    src_root_folder = "/data/siyu.liu/datasets/FF++_c40/images-f/"  # 替换为你的源根文件夹路径
    dest_root_folder = "/data/siyu.liu/datasets/FF++_c40/images1/"  # 替换为你的目标根文件夹路径
    process_all_subfolders(src_root_folder, dest_root_folder)
