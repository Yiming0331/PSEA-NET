"""
Evaluates a folder of video files or a single file with a xception binary
xception是视频二元分类网(真/假)
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
Modified to use MTCNN and continuous naming logic.
"""
import sys, os
import argparse
from os.path import join
sys.path.insert(0, os.getcwd())
import cv2
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm
import numpy as np
from pathlib import Path

# 引入 MTCNN
from facenet_pytorch import MTCNN 
from network.models import model_selection
from dataset.transform import xception_default_data_transforms

def get_boundingbox(box, width, height, scale=1.3, minsize=None):
    """
    Expects a list/array of [x1, y1, x2, y2] to generate a quadratic bounding box.
    :param box: [x1, y1, x2, y2] coordinates from MTCNN
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region (default 1.3 means +30%)
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    # MTCNN 返回的坐标可能是浮点数，转为整数
    x1 = int(box[0])
    y1 = int(box[1])
    x2 = int(box[2])
    y2 = int(box[3])
    
    # 计算最大边长，并乘以缩放比例 (默认1.3，即扩大30%)
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
            
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    # Check for too big bb size for given x, y
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    return x1, y1, size_bb

def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    """
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image

def test_full_image_network(video_path, output_path,
                            start_frame=0, end_frame=None, cuda=True):
    """
    Reads a video and evaluates a subset of frames.
    Outputs are only given if a face is present.
    """
    print('Starting: {}'.format(video_path))
    
    reader = cv2.VideoCapture(video_path)
    
    os.makedirs(output_path, exist_ok=True)
    
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # 初始化 MTCNN
    # keep_all=True: 允许检测多张脸 (虽然我们只取第一张)
    device = torch.device('cuda:0' if cuda and torch.cuda.is_available() else 'cpu')
    face_detector = MTCNN(keep_all=True, device=device)

    # Frame numbers logic
    frame_num = 0
    
    # 如果未指定 end_frame，默认处理到视频结束
    if end_frame is None:
        end_frame = num_frames
    else:
        # 防止输入的 end_frame 超过视频实际长度
        end_frame = min(end_frame, num_frames)

    print("Start frame:", start_frame)
    print("End frame:", end_frame)
    
    pbar = tqdm(total=end_frame-start_frame)

    # 这是一个独立的计数器，用于给保存的图片连续命名 (000.jpg, 001.jpg...)
    save_count = 0 

    while reader.isOpened():
        _, image = reader.read()
        if image is None:
            break

        # 如果还没到开始帧，跳过
        if frame_num < start_frame:
            frame_num += 1
            continue
        
        # 如果超过了结束帧，退出循环
        if frame_num >= end_frame:
            break

        pbar.update(1)

        height, width = image.shape[:2]

        # 1. 转换颜色空间给 MTCNN 用 (BGR -> RGB -> PIL)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        pil_img = pil_image.fromarray(image_rgb)
        
        try:
            # 2. 检测人脸
            boxes, _ = face_detector.detect(pil_img)
        except Exception as e:
            # 极少数情况 MTCNN 可能会报错，做个保护
            boxes = None

        # 3. 判断逻辑：
        # 如果有人脸 -> 保存图片 -> save_count + 1
        # 如果没人脸 -> 不保存 -> save_count 不变
        if boxes is not None and len(boxes) > 0:
            # 取第一个人脸
            face_box = boxes[0]
            
            # 获取裁剪坐标 (包含 scale=1.3 的放大逻辑)
            x, y, size = get_boundingbox(face_box, width, height)
            
            # 构造文件名，使用 save_count 保证连续性
            save_name = os.path.join(output_path, f'{save_count:03}.jpg')
            
            # 使用原始 image (BGR) 进行裁剪保存，保证颜色正确
            cv2.imwrite(save_name, image[y:y+size, x:x+size])
            
            # 只有成功保存了，序号才加 1
            save_count += 1

        # 原始视频帧号正常递增
        frame_num += 1
        
    pbar.close()
    reader.release()
    print(f'Finished! Output saved under {output_path}, Total images extracted: {save_count}')

    

    #  如果这个数据集非常大，容易网络中断我们会搞一个标记txt文件，证明这个视频搞完了
    # 【新增】处理完全部帧后，写入一个标记文件
    
    done_file_path = os.path.join(output_path, 'done.txt')
    with open(done_file_path, 'w') as f:
        f.write('finished')


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str, default='/data/siyu.liu/datasets/Celeb-DF-v3/Celeb-synthesis/FaceReenact/LivePortrait/')
    p.add_argument('--output_path', '-o', type=str,
                   default='/data/siyu.liu/datasets/Celeb-DF-v3/Celeb_image/Celeb-synthesis/FaceReenact/')
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    p.add_argument('--cuda', type=bool, default=True, help='use cuda')
    args = p.parse_args()

    video_dir = Path(args.video_path).resolve()
    output_dir = Path(args.output_path).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 递归查找所有mp4文件
    for file in video_dir.rglob('*.mp4'):
        file_path = file.as_posix()
        temp_path = file.relative_to(video_dir)
        save_dir = output_dir / temp_path
        save_dir = save_dir.parent / save_dir.stem

        save_dir = output_dir / video_dir.name / temp_path.parent / file.stem

        # 该方法有些烂，但是也可以检查：如果文件夹存在，并且里面有图片（说明之前跑过了），就跳过
        # if save_dir.exists() and any(save_dir.iterdir()):
        #     print(f"Skipping {save_dir} (Already processed)")
        #     continue
        
        # 【修改后】检查是否存在 done.txt
        done_flag = save_dir / 'done.txt'
        
        if save_dir.exists() and done_flag.exists():
            print(f"Skipping {save_dir} (Already processed completely)")
            continue

        print(f"Processing: {file_path} -> {save_dir}")
        

        # 如果文件夹存在但没有 done.txt，说明上次没跑完
        # 可以选择清空文件夹重新跑，或者接着跑（取决于你的覆盖逻辑）
        # 你的代码使用的是 save_count 覆盖命名，建议如果没跑完就清空重跑，防止序号混乱
        # 打开这个代码的时候需要给上面一段存在图片就跳过那段代码给它注释了
        # 且需要给  def test_full_image_network() 该方法最下面的那几个东西给她打开

        if save_dir.exists() and not done_flag.exists():
            print(f"Found incomplete folder {save_dir}, removing and reprocessing...")
            import shutil
            shutil.rmtree(save_dir) # 删掉残缺的文件夹


        # 构造参数并调用
        # 注意：为了避免多线程/循环中 args 对象被污染，最好每次拷贝或者直接传参
        # 这里因为是同步顺序执行，直接修改 args 也是可以的
        current_args = argparse.Namespace(**vars(args))
        current_args.video_path = file_path
        current_args.output_path = str(save_dir)
        
        test_full_image_network(**vars(current_args))