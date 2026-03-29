import cv2
import os


def video_to_images_by_frame(video_path, output_dir, frame_interval=5):
    """
    将视频按帧间隔切成图片

    参数：
        video_path: 输入视频路径
        output_dir: 输出图片文件夹
        frame_interval: 每隔多少帧保存一张
    """
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"找不到视频文件: {video_path}")

    os.makedirs(output_dir, exist_ok=True)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise ValueError("视频打开失败，请检查文件格式或路径是否正确")

    frame_count = 0
    saved_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frame_interval == 0:
            image_name = os.path.join(output_dir, f"frame_{saved_count:05d}.jpg")
            cv2.imwrite(image_name, frame)
            saved_count += 1

        frame_count += 1

    cap.release()
    print(f"处理完成，共读取 {frame_count} 帧，保存 {saved_count} 张图片")


if __name__ == "__main__":
    video_to_images_by_frame(
        video_path="input.mp4",
        output_dir="output_images",
        frame_interval=5
    )