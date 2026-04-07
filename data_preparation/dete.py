import os
import cv2

def process_videos_in_folder(input_folder, output_folder):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    for root, dirs, files in os.walk(input_folder):
        for file in files:
            if file.endswith(('.mp4')):  # 添加更多的视频格式
                video_path = os.path.join(root, file)
                video_name = os.path.splitext(file)[0]
                
                # 构建与视频路径相同的输出路径
                relative_path = os.path.relpath(root, input_folder)
                output_video_folder = os.path.join(output_folder, relative_path, video_name)
                
                if not os.path.exists(output_video_folder):
                    os.makedirs(output_video_folder)
                
                cap = cv2.VideoCapture(video_path)
                frame_count = 0
                
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    
                    frame_filename = os.path.join(output_video_folder, f"{frame_count:03d}.jpg")
                    cv2.imwrite(frame_filename, frame)
                    frame_count += 1
                
                cap.release()
                print(f"Processed {video_name} with {frame_count} frames.")
                
    print("All videos processed.")

if __name__ == "__main__":
    input_folder = "/data/siyu.liu/datasets/Celeb-DF-v2/"  # 替换为你的视频文件夹路径
    output_folder = "/data/siyu.liu/datasets/Celeb-DF-v2/images-p"  # 替换为你想要保存图像的文件夹路径
    process_videos_in_folder(input_folder, output_folder)


