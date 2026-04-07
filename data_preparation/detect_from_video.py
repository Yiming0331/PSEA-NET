"""
Evaluates a folder of video files or a single file with a xception binary
classification network.

Usage:
python detect_from_video.py
    -i <folder with video files or path to video file>
    -m <path to model file>
    -o <path to output folder, will write one or multiple output videos there>

Author: Andreas Rössler
"""
# 这是一个文档字符串（docstring），解释了这个脚本的用途、用法和作者。
# 它告诉你这个脚本是用来评估视频（或视频文件夹）的，使用的是一个 Xception 分类网络。

# --- 导入库 ---
import os  # 导入操作系统库，用于处理文件和目录路径。
import argparse  # 导入参数解析库，用于从命令行读取用户输入的参数（如视频路径、模型路径等）。
from os.path import join  # 从 os 库中只导入 join 函数，它用于安全地拼接文件路径（例如 folder + filename）。
import cv2  # 导入 OpenCV 库，这是处理图像和视频的核心库（读取视频、显示图像、绘制矩形等）。
import dlib  # 导入 dlib 库，这里专门用于人脸检测。
import torch  # 导入 PyTorch 库，用于加载模型和执行深度学习推理。
import torch.nn as nn  # 导入 PyTorch 的神经网络模块，这里用到了 nn.Softmax。
from PIL import Image as pil_image  # 导入 PIL (Python Imaging Library) 库，并给它取个别名 pil_image。它用于在 OpenCV 图像和 PyTorch 变换（transform）之间进行格式转换。
from tqdm import tqdm  # 导入 tqdm 库，用于在处理视频帧时显示一个漂亮的进度条。

# 从同一目录下的 network/models.py 文件中导入 model_selection 函数。
# 这个函数的作用是根据你给的名字（比如 'xception'）来创建对应的模型对象。
from network.models import model_selection
# 从 dataset/transform.py 文件中导入一个字典。
# 这个字典定义了在模型推理前需要对图像做什么样的预处理（比如缩放、归一化）。
from dataset.transform import xception_default_data_transforms


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    """
    Expects a dlib face to generate a quadratic bounding box.
    :param face: dlib face class (dlib 检测到的人脸对象)
    :param width: frame width (原始帧的宽度)
    :param height: frame height (原始帧的高度)
    :param scale: bounding box size multiplier to get a bigger face region (边界框缩放比例，默认1.3倍)
    :param minsize: set minimum bounding box size (最小边界框尺寸)
    :return: x, y, bounding_box_size in opencv form (返回正方形边界框的左上角x, y坐标和边长size)
    """
    # dlib 的 face 对象包含 .left(), .top(), .right(), .bottom() 方法来获取人脸框的四个角点坐标。
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()
    
    # 计算人脸框的宽度(x2-x1)和高度(y2-y1)，取其中较大者，并乘以缩放比例 scale（默认1.3）。
    # 这样可以确保得到一个比 dlib 原始框更大的、能容纳整个人脸的区域。
    # int() 将结果转为整数。
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    
    # 如果设置了 minsize（最小尺寸）
    if minsize:
        # 检查计算出的 size_bb 是否小于 minsize
        if size_bb < minsize:
            # 如果是，则强制使用 minsize 作为边长
            size_bb = minsize
            
    # 计算人脸框的中心点坐标
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # --- 边界检查 ---
    # 以中心点为基准，计算出新的正方形框的左上角坐标 x1, y1
    # int(center_x - size_bb // 2) 是理论上的 x1
    # max(..., 0) 是为了防止这个坐标小于0，即超出图像的左边界或上边界。
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    
    # 再次检查边长，确保边长 `size_bb` 不会太大导致框超出图像的右边界 (width - x1) 或下边界 (height - y1)。
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)

    # 返回这个经过修正和放大的正方形框的左上角坐标 x1, y1 和它的边长 size_bb
    return x1, y1, size_bb


def preprocess_image(image, cuda=True):
    """
    Preprocesses the image such that it can be fed into our network.
    During this process we envoke PIL to cast it into a PIL image.

    :param image: numpy image in opencv form (i.e., BGR and of shape [H, W, C])
    :return: pytorch tensor of shape [1, 3, image_size, image_size], not
    necessarily casted to cuda
    """
    # --- 颜色通道转换 ---
    # Revert from BGR
    # OpenCV 读取的图像默认是 BGR 颜色通道顺序。
    # PIL 和 PyTorch 通常期望 RGB 顺序，所以在这里进行转换。
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # --- 图像预处理 ---
    # Preprocess using the preprocessing function used during training and
    # casting it to PIL image
    # 从之前导入的 xception_default_data_transforms 字典中，获取 'test' 键对应的预处理流程。
    # 这个流程在 dataset/transform.py 中定义，通常包括：
    # 1. transforms.Resize((299, 299))  -- 缩放到 299x299，因为 Xception 模型需要这个尺寸
    # 2. transforms.ToTensor()          -- 转换为 PyTorch 张量
    # 3. transforms.Normalize([0.5]*3, [0.5]*3) -- 归一化到 [-1, 1] 范围
    preprocess = xception_default_data_transforms['test']
    
    # pil_image.fromarray(image) 先将 NumPy 数组（OpenCV 图像）转为 PIL 图像。
    # 然后 preprocess(...) 应用上面定义的所有预处理步骤。
    preprocessed_image = preprocess(pil_image.fromarray(image))
    
    # --- 增加 Batch 维度 ---
    # Add first dimension as the network expects a batch
    # 模型的输入总是批量的（batch）。
    # preprocessed_image 此时的形状是 [C, H, W] (例如 [3, 299, 299])。
    # .unsqueeze(0) 在最前面增加一个维度，使其形状变为 [1, C, H, W] (例如 [1, 3, 299, 299])。
    preprocessed_image = preprocessed_image.unsqueeze(0)
    
    # --- 转移到 GPU ---
    if cuda:
        # 如果 cuda 参数为 True（即命令行中指定了 --cuda），则将张量移动到 GPU 上。
        preprocessed_image = preprocessed_image.cuda()
        
    return preprocessed_image


def predict_with_model(image, model, post_function=nn.Softmax(dim=1),
                       cuda=True):
    """
    Predicts the label of an input image. Preprocesses the input image and
    casts it to cuda if required

    :param image: numpy image (传入的是裁剪后的人脸图像，NumPy 格式)
    :param model: torch model with linear layer at the end (加载好的 PyTorch 模型)
    :param post_function: e.g., softmax (后处理函数，默认使用 Softmax 将输出转为概率)
    :param cuda: enables cuda, must be the same parameter as the model (是否使用 GPU)
    :return: prediction (1 = fake, 0 = real) (返回整数预测：0 或 1) 和 output (原始概率输出)
    """
    # --- 1. 预处理 ---
    # Preprocess
    # 调用上一个函数，将传入的 NumPy 图像转换为模型所需的、已归一化的、带 batch 维度的 GPU 张量。
    preprocessed_image = preprocess_image(image, cuda)

    # --- 2. 模型推理 ---
    # Model prediction
    # 这是核心的推理步骤。将预处理好的图像张量送入模型 `model`。
    # `output` 是模型的原始输出，通常称为 logits，不是 0-1 之间的概率。
    output = model(preprocessed_image)
    # 应用 `post_function`（默认为 Softmax），将 logits 转换成 0 到 1 之间的概率值。
    # `dim=1` 表示在类别维度上进行 Softmax。
    output = post_function(output)

    # --- 3. 获取最终预测 ---
    # Cast to desired
    # `torch.max(output, 1)` 会在维度 1 (类别维度) 上寻找最大值。
    # 它返回两个值：(最大概率值, 最大概率值所在的索引)
    # 我们用 `_` 忽略最大概率值，只保留索引，存入 `prediction` 变量。
    # 这个 `prediction` 就是模型的预测结果（例如 0 或 1）。
    _, prediction = torch.max(output, 1)    # argmax
    
    # 将 `prediction`（它是一个 PyTorch 张量）从 GPU 转移回 CPU (`.cpu()`)，
    # 再转为 NumPy 数组 (`.numpy()`)，然后取 `float()` 值。
    prediction = float(prediction.cpu().numpy())

    # 返回整数型的预测结果 (0 或 1) 和包含两个概率的原始输出张量 `output`。
    return int(prediction), output


def test_full_image_network(video_path, model_path, output_path,
                            start_frame=0, end_frame=None, cuda=True):
    """
    Reads a video and evaluates a subset of frames with the a detection network
    that takes in a full frame. Outputs are only given if a face is present
    and the face is highlighted using dlib.
    :param video_path: path to video file (视频文件路径)
    :param model_path: path to model file (should expect the full sized image) (模型权重文件路径 .p)
    :param output_path: path where the output video is stored (输出路径)
    :param start_frame: first frame to evaluate (开始帧)
    :param end_frame: last frame to evaluate (结束帧)
    :param cuda: enable cuda (是否使用 GPU)
    :return:
    """
    print('Starting: {}'.format(video_path))

    # --- 1. 准备视频读取和写入 ---
    # Read and write
    # 使用 OpenCV 打开视频文件，创建 `reader` 对象用于逐帧读取。
    reader = cv2.VideoCapture(video_path)

    # 构造输出视频的文件名，例如 "my_video.mp4" -> "my_video.avi"
    video_fn = video_path.split('/')[-1].split('.')[0]+'.avi'
    # 创建输出文件夹（如果它不存在）。`exist_ok=True` 表示如果文件夹已存在，也不会报错。
    os.makedirs(output_path, exist_ok=True)
    # 定义视频的编码格式（这里是 MJPG）。
    fourcc = cv2.VideoWriter_fourcc(*'MJPG')
    # 从 `reader` 对象获取原视频的帧率 (fps) 和总帧数 (num_frames)。
    fps = reader.get(cv2.CAP_PROP_FPS)
    num_frames = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
    # 先将视频写入器 `writer` 初始化为 None。
    # 我们稍后在知道视频帧的宽高后才会真正创建它。
    writer = None

    # --- 2. 初始化人脸检测器 ---
    # Face detector
    # 初始化 dlib 的人脸检测器。
    face_detector = dlib.get_frontal_face_detector()

    # --- 3. 加载模型 ---
    # Load model
    # 调用 `model_selection` 函数，创建 Xception 模型的**架构**（骨架），
    # 指定输出类别为 2 (real/fake)。
    model, *_ = model_selection(modelname='xception', num_out_classes=2)
    
    # 检查是否提供了模型路径
    if model_path is not None:
        # 使用 `torch.load` **加载预训练的模型权重文件**（.p 文件）。
        # 这是最关键的模型加载步骤，它将权重填充到上面创建的 `model` 架构中。
        model = torch.load(model_path)
        print('Model found in {}'.format(model_path))
    else:
        # 如果没有提供模型路径，则使用一个随机初始化的模型（通常用于测试）。
        print('No model found, initializing random model.')
        
    # 如果 `cuda` 为 True (即命令行传入了 --cuda)，则将整个模型移动到 GPU。
    if cuda:
        model = model.cuda()

    # --- 4. 准备绘制参数和进度条 ---
    # Text variables
    # 定义稍后在视频上写字（"real" / "fake"）时要用的字体、粗细、大小。
    font_face = cv2.FONT_HERSHEY_SIMPLEX
    thickness = 2
    font_scale = 1

    # Frame numbers and length of output video
    frame_num = 0  # 初始化帧计数器
    # 断言（assert）：确保开始帧号小于总帧数，否则程序会报错。
    assert start_frame < num_frames - 1
    # 设置结束帧。如果用户没有指定 `end_frame` (即 `end_frame` 为 None)，
    # 则 `end_frame` 会被设置为视频的总帧数 `num_frames`，表示处理到最后。
    end_frame = end_frame if end_frame else num_frames
    # 使用 `tqdm` 创建一个进度条，总长度为 (end_frame - start_frame)。
    pbar = tqdm(total=end_frame-start_frame)

    # --- 5. 开始逐帧处理循环 ---
    while reader.isOpened():  # 循环，只要视频打开就一直执行。
        # `reader.read()` 读取一帧图像。
        # `_` 存储读取是否成功的布尔值（这里忽略），`image` 存储读取到的图像帧（NumPy 数组）。
        _, image = reader.read()
        
        # 如果 `image` 为 `None`，表示视频已经读取完毕。
        if image is None:
            break  # 跳出 while 循环
            
        frame_num += 1  # 帧计数器加 1。

        # 如果当前帧号小于指定的 `start_frame`，则跳过本次循环，处理下一帧。
        if frame_num < start_frame:
            continue
            
        pbar.update(1)  # 更新进度条，表示已处理一帧。

        # Image size
        # 获取当前帧的高度和宽度，`image.shape` 返回 (height, width, channels)。
        height, width = image.shape[:2]

        # Init output writer
        # 检查 `writer` 是否还是 None（即是否是第一帧被处理）。
        if writer is None:
            # 如果是，则使用 OpenCV 的 `VideoWriter` 创建一个视频写入器实例。
            # 需要传入：输出路径、编码格式、帧率、以及帧的宽高。
            # `(height, width)[::-1]` 是一个技巧，将 (height, width) 转换为 (width, height)，
            # 因为 VideoWriter 需要 (width, height) 格式。
            writer = cv2.VideoWriter(join(output_path, video_fn), fourcc, fps,
                                     (height, width)[::-1])

        # --- 6. 人脸检测 ---
        # 2. Detect with dlib
        # 将彩色图像 `image` 转换为灰度图像 `gray`，因为 dlib 人脸检测器需要灰度图。
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # `face_detector(gray, 1)` 在灰度图上执行人脸检测。
        # `faces` 是一个包含所有检测到的人脸的列表（dlib 对象）。
        faces = face_detector(gray, 1)
        
        # 检查 `faces` 列表的长度，即是否检测到了至少一张人脸。
        if len(faces):
            # For now only take biggest face
            # 如果检测到人脸，只取列表中的第一个 `faces[0]` 进行处理。
            # 注意：这不一定是最大的人脸，只是 dlib 返回的第一个。
            face = faces[0]

            # --- 7. 人脸裁剪与预测 ---
            # --- Prediction ---------------------------------------------------
            # Face crop with dlib and bounding box scale enlargement
            # 调用我们之前定义的 `get_boundingbox` 函数，
            # 传入 dlib 人脸对象和帧的宽高，获取放大后的正方形人脸框 (x, y, size)。
            x, y, size = get_boundingbox(face, width, height)
            
            # 使用 NumPy 的切片功能，从原始彩色图像 `image` 中裁剪出人脸区域。
            # `y:y+size` 是高度范围，`x:x+size` 是宽度范围。
            cropped_face = image[y:y+size, x:x+size]
            

            # Actual prediction using our model
            # **调用核心预测函数**
            # 将裁剪出的人脸 `cropped_face` 和加载好的 `model` 传入。
            # `prediction` 将是 0 (real) 或 1 (fake)。
            # `output` 是原始的概率张量。
            prediction, output = predict_with_model(cropped_face, model,
                                                    cuda=cuda)
            # ------------------------------------------------------------------

            # --- 8. 在图像上绘制结果 ---
            # Text and bb
            # 获取 dlib **原始**人脸框的坐标和宽高，我们将用这个原始框来绘制。
            x = face.left()
            y = face.top()
            w = face.right() - x
            h = face.bottom() - y
            
            # 根据预测结果（0 或 1）设置文本标签 `label`。
            label = 'fake' if prediction == 1 else 'real'
            
            # 设置框和文字的颜色。如果是 real (0)，则为绿色 (0, 255, 0) (BGR)；
            # 如果是 fake (1)，则为红色 (0, 0, 255) (BGR)。
            color = (0, 255, 0) if prediction == 0 else (0, 0, 255)
            
            # 将 `output` 概率张量转为列表（例如 [0.05, 0.95]），并格式化为保留两位小数的字符串。
            output_list = ['{0:.2f}'.format(float(x)) for x in
                           output.detach().cpu().numpy()[0]]
                           
            # `cv2.putText`：在图像上绘制文本。
            # 参数：图像, 文本内容, 坐标 (x, y+h+30 即框的下面), 字体, 大小, 颜色, 粗细, 线条类型
            cv2.putText(image, str(output_list)+'=>'+label, (x, y+h+30),
                        font_face, font_scale,
                        color, thickness, 2)
            
            # `cv2.rectangle`：在图像上绘制矩形框。
            # 参数：图像, 左上角坐标 (x, y), 右下角坐标 (x+w, y+h), 颜色, 粗细
            cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)

        # 检查是否达到了设定的 `end_frame`。
        if frame_num >= end_frame:
            break  # 如果达到，则停止循环。

        # --- 9. 显示与保存（可选） ---
        # Show
        # `cv2.imshow('test', image)`：打开一个名为 'test' 的窗口，实时显示处理后（带有框和标签）的图像帧。
        cv2.imshow('test', image)
        
        # `cv2.waitKey(33)`：等待 33 毫秒。
        # 这是为了让 `imshow` 有时间刷新窗口，否则窗口会卡住。
        # 33 毫秒大约对应 30 fps (1000 / 30 ≈ 33)。
        cv2.waitKey(33)     # About 30 fps
        
        # `writer.write(image)`：将处理后的帧写入到视频文件。
        # **注意：在原始代码中，这一行被注释掉了！**
        # 这意味着，**默认情况下，这个脚本不会把带框的视频保存到文件**，只会实时显示。
        # 如果你想保存视频，需要取消这一行的注释。
        # writer.write(image)
        
    # --- 10. 循环结束，清理 ---
    pbar.close()  # 关闭进度条。
    
    # 检查 `writer` 是否被创建（即是否至少处理了一帧）。
    if writer is not None:
        writer.release()  # 释放视频写入器，完成文件写入。
        print('Finished! Output saved under {}'.format(output_path))
    else:
        # 如果 `writer` 仍然是 None（可能是因为视频文件为空或 `start_frame` 设置不当）。
        print('Input video file was empty')


# --- 脚本入口点 ---
if __name__ == '__main__':
    # `if __name__ == '__main__':` 是 Python 脚本的入口点。
    # 当你通过 `python detect_from_video.py` 运行此文件时，会从这里开始执行。
    
    # --- 1. 定义命令行参数 ---
    # 创建一个 `ArgumentParser` 对象，用于解析命令行参数。
    p = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
        
    # 定义脚本接受的命令行参数：
    # `--video_path` (或 `-i`)：输入视频的路径（或文件夹路径）。
    # `default` 设置了如果用户不提供此参数时的默认值。
    p.add_argument('--video_path', '-i', type=str,default='/data/hainan3_data/dataset/FF++')
    
    # `--model_path` (或 `-mi`)：预训练模型 `full_c23.p` 的路径。
    p.add_argument('--model_path', '-mi', type=str, default='/data/yiming.hao/code/TALL4Deepfake_raw_2_new/faceforensics++_models_subset/full/xception/full_c23.p')
    
    # `--output_path` (或 `-o`)：输出路径（用于保存视频，如果取消注释的话）。
    p.add_argument('--output_path', '-o', type=str,
                   default='./ou')
                   
    # `--start_frame` / `--end_frame`：可选，用于指定只处理视频的某一段。
    p.add_argument('--start_frame', type=int, default=0)
    p.add_argument('--end_frame', type=int, default=None)
    
    # `--cuda`：一个开关（action='store_true'）。
    # 如果在命令行中加入 `--cuda`，则 `args.cuda` 的值为 True。
    # 如果不加，`args.cuda` 的值为 False。
    p.add_argument('--cuda', action='store_true')
    
    # --- 2. 解析参数 ---
    # `p.parse_args()`：解析命令行输入的参数，并将它们存入 `args` 对象。
    # 例如，可以通过 `args.video_path` 来访问视频路径。
    args = p.parse_args()

    # --- 3. 执行主逻辑 ---
    video_path = args.video_path  # 获取视频路径
    
    # 检查 `video_path` 是不是一个以 '.mp4' 或 '.avi' 结尾的**文件**。
    if video_path.endswith('.mp4') or video_path.endswith('.avi'):
        # 如果是单个文件，则直接调用 `test_full_image_network` 函数来处理这个文件。
        # `**vars(args)` 是一种将 `args` 对象（包含了所有参数）解包并作为关键字参数传递给函数的方式。
        # 相当于调用：test_full_image_network(video_path=args.video_path, model_path=args.model_path, ...)
        test_full_image_network(**vars(args))
    else:
        # 如果**不是**单个文件（即它是一个**文件夹**），则：
        # `os.listdir(video_path)` 列出该文件夹下的所有文件名。
        videos = os.listdir(video_path)
        # 循环遍历文件夹里的每一个文件名 `video`。
        for video in videos:
            # 更新 `args` 对象中的视频路径，使其指向当前循环到的视频文件。
            # `join` 会正确地拼接路径，例如 "folder/" + "video.mp4" -> "folder/video.mp4"
            args.video_path = join(video_path, video)
            # 再次调用主函数，传入更新后的 `args` 来处理这个视频。
            test_full_image_network(**vars(args))