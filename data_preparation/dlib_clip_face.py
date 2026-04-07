"""
Evaluates a folder of video files or a single file with a xception binary
xception是视频二元分类网（真/假）
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""
import sys,os
import argparse
from os.path import join
sys.path.insert(0,os.getcwd())
import cv2
import dlib
import torch
import torch.nn as nn
from PIL import Image as pil_image
from tqdm import tqdm

from network.models import model_selection
from dataset.transform import xception_default_data_transforms
#from facenet_pytorch import MTCNN


#这个函数用于根据传入的 dlib 人脸对象 (face) 生成一个正方形的边界框。avi
#它的输入参数包括帧的宽度 (width) 和高度 (height)，
#以及一个可选参数 scale 用于调整边界框的大小以获取更大的人脸区域，还有一个可选参数 minsize 用于设置最小边界框的大小。
def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param scale: bounding box size multiplier to get a bigger face region
    :param minsize: set minimum bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face.left()#(x1,y1是人脸的左上角坐标)
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()#(x1,y1是人脸的右下角坐标)
    size_bb = int(max(x2 - x1, y2 - y1) * scale)#并乘以 scale 参数来调整边界框大小
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
#函数返回的是边界框的左上角坐标 (x, y) 以及边界框的大小，以适应 OpenCV 的格式。



#使用 PIL 将图像转换为 PIL 图像，期望输入的形状是 [batch_size, channels, height, width]
def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # Revert from BGR
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    preprocess = xception_default_data_transforms['test']
    preprocessed_image = preprocess(pil_image.fromarray(image))
    # Add first dimension as the network expects a batch
    preprocessed_image = preprocessed_image.unsqueeze(0)
    if cuda:
        preprocessed_image = preprocessed_image.cuda()
    return preprocessed_image





#读取视频，并在其中使用指定的模型对一部分帧进行评估，输出结果保存在指定路径下的视频文件中
def test_full_image_network(video_path,output_path,
                            start_frame=0, end_frame=None, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file
    :param model_path: path to model file (should expect the full sized image)
    :param output_path: path where the output video is stored
    :param start_frame: first frame to evaluate
    :param end_frame: last frame to evaluate
    :param cuda: enable cuda
    :return:
    """
    print('Starting: {}'.format(video_path))
    # Read and write
    #读取指定路径下的视频，并将其转换为AVI格式的视频文件
    reader = cv2.VideoCapture(video_path)
    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'

    #输出目录
    os.makedirs(output_path, exist_ok=True)
    #定义了视频编码器
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')


    #获取输入视频的帧率
    fps = reader.get(cv2.CAP_PROP_FPS)

    #获取输入视频的帧数
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    writer = None

    # Face detector
    face_detector = dlib.get_frontal_face_detector()
    #face_detector= MTCNN(keep_all=True)

    # Text variables
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0
    print("Start frame:", start_frame)
    print("Number of frames:", num_frames)
    assert start_frame < num_frames - 1
    end_frame = end_frame if end_frame else num_frames
    pbar = tqdm(total=end_frame-start_frame)

   
   #逐帧读取视频
    while reader.isOpened():
        #read()函数返回两个值，第一个值是一个布尔值，表示是否成功读取到帧，第二个值是读取到的帧图像
        _, image = reader.read()
        if image is None:
            break
        # frame_num += 1#处理的帧数加1

        if frame_num < start_frame:
            continue
        pbar.update(1)

        # Image size
        height, width = image.shape[:2]#获取当前帧图像的高度和宽度。

        # # Init output writer
        # if writer is None:
        # #创建一个视频写入器，包括输出视频文件的路径，视频编码器的四字符码fourcc，帧率fps，以及视频的尺寸(height, width)
        #     writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
        #                              (height, width)[::-1])

        # 2. Detect with dlib
        #dlib检测人脸，并对检测到的人脸进行操作和预测
            
        #将彩色图像转换为灰度图像，以便进行人脸检测
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        #人脸检测器face_detector
        faces = face_detector(gray, 1)

        #如果检测到人脸，取第一个检测到的人脸作为操作对象
        if len(faces):
            # For now only take biggest face
            face = faces[0]


            #调用get_boundingbox函数获取人脸区域的边界框
            x, y, size = get_boundingbox(face, width, height)


            cv2.imwrite(os.path.join(output_path, f'{frame_num:03}.jpg'),image[y:y+size, x:x+size])
        if frame_num >= end_frame:
            break
        # save img
        # video_name = os.path.splitext(os.path.basename(video_path))[0]
        # 这个代码放在这里有可能没有人脸时候会报错
        # cv2.imwrite(os.path.join(output_path, f'{frame_num:03}.jpg'),image[y:y+size, x:x+size])

        # Show
        # cv2.imshow('test', image)
        # cv2.waitKey(33)     # About 30 fps
        # writer.write(image)
        frame_num += 1#处理的帧数加1
    pbar.close()
    if writer is not None:
        writer.release()
        print('Finished! Output saved under {}'.format(output_path))
    else:
        print('Input video file was empty')


if __name__ == '__main__':
    from pathlib import Path
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--video_path', '-i', type=str,default='/data/siyu.liu/datasets/SDFVD/videos_real/')
    # p.add_argument('--model_path', '-mi', type=str, default='/data/siyu.liu/code/TALL4Deepfake/data_preparation/swin_base_patch4_window7_224_22k.pth')
    p.add_argument('--output_path', '-o', type=str,
                   default='/data/siyu.liu/datasets/SDFVD/image_real/')
    p.add_argument('--start_frame', type=int, default=0)#0改1
    p.add_argument('--end_frame', type=int, default=None)#None改成300
    p.add_argument('--cuda', type=bool, default=True,  help='use cuda')
    args = p.parse_args()


#给定的视频路径处理单个视频文件或一个目录中的所有视频文件
    video_dir = Path(args.video_path).resolve()
    output_dir =  Path(args.output_path).resolve()
    output_dir.mkdir(parents=True,exist_ok=True)
    test_num = 0
    for file in video_dir.rglob('*.mp4'):
        file_path = file.as_posix()
        temp_path = file.relative_to(video_dir)
        save_dir = output_dir / temp_path
        save_dir = save_dir.parent / save_dir.stem

        # ===【在这里添加这段代码】===
        # 检查：如果文件夹存在，并且里面有图片（说明之前跑过了），就跳过
        if save_dir.exists() and any(save_dir.iterdir()):
            print(f"Skipping {save_dir} (Already processed)")
            continue
        # ==========================

        print(file_path,save_dir)
        args.video_path = file_path
        args.output_path = save_dir
        test_full_image_network(**vars(args))

    # video_dir = args.video_path
    # output_dir = args.output_path
    # os.makedirs(output_dir,exist_ok=True)
    # test_num = 0
    # for root, dirs, files in os.walk(video_dir):
    #     save_dir = None
    #     for file in files:
    #         if file.endswith('.mp4'):
    #             if test_num>2:
    #                 break
    #             file_path = os.path.join(root,file)
    #             temp_path = os.path.relpath(root, video_dir)
    #             save_dir = os.path.join(output_dir,temp_path)
    #             save_dir = os.path.join(save_dir,os.path.splitext(file)[0])
    #             print(file_path,save_dir)
    #             args.video_path = file_path
    #             args.output_path = save_dir
    #             test_full_image_network(**vars(args))

