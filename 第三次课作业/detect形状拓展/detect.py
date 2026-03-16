import cv2
import numpy as np

# ----------------- 1. 初始化设置 -----------------
# 【修改处 1】：不再使用 Socket，而是调用本地默认摄像头 (编号 0)
cap = cv2.VideoCapture(0)

# 检查摄像头是否成功打开
if not cap.isOpened():
    print("错误：无法打开摄像头！")
    exit()

# 加载人脸识别模型 
# 注意：请确保 'haarcascade_frontalface_default.xml' 文件和本 Python 文件在同一个文件夹下！
faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 使用 HSV 色彩空间提取红色
lower_red1 = np.array([0, 120, 70])
upper_red1 = np.array([10, 255, 255])
lower_red2 = np.array([170, 120, 70])
upper_red2 = np.array([180, 255, 255])

print("按 'q' 键退出程序")

while True:
    # ----------------- 2. 获取视频流 -----------------
    # 【修改处 2】：从摄像头读取一帧画面
    ret, frame = cap.read()
    
    # 如果读取失败，退出循环
    if not ret:
        print("无法获取画面")
        break

    frame_copy = frame.copy()
    
    # 定义机器人动作状态标记
    robot_action = "SEARCH" # 默认动作：搜索
    found_face = False
    found_red_card = False
    found_red_ball = False

    # ----------------- 3. 人脸检测 -----------------
    faces_rects = faceCascade.detectMultiScale(
        frame, scaleFactor=1.05, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE
    )
    for (x, y, w, h) in faces_rects:
        cv2.rectangle(frame_copy, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(frame_copy, "Face", (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        found_face = True # 发现人脸！

    # ----------------- 4. 红色物体检测 (球 / 牌) -----------------
    # 将 BGR 转为 HSV
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # 提取红色掩膜并合并
    mask1 = cv2.inRange(hsv_frame, lower_red1, upper_red1)
    mask2 = cv2.inRange(hsv_frame, lower_red2, upper_red2)
    mask = cv2.bitwise_or(mask1, mask2)
    
    # 开运算去除小噪点 
    kernel = np.ones((5, 5), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    # 寻找轮廓
    contours, hierarchy = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # 遍历所有找到的红色轮廓
    for cnt in contours:
        area = cv2.contourArea(cnt)
        
        # 过滤掉面积太小的噪点 (比如小于 500 像素的不处理)
        if area < 500:
            continue
            
        # 计算轮廓周长和近似多边形
        perim_px = cv2.arcLength(cnt, True)
        epsilon = 0.04 * perim_px 
        approx = cv2.approxPolyDP(cnt, epsilon, True)

        # 绘制轮廓 
        cv2.drawContours(frame_copy, [approx], 0, (0, 255, 255), 2)

        # 计算重心 (中心点 cx, cy)
        M = cv2.moments(cnt)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            cx, cy = 0, 0

        # 判断形状
        sides = len(approx)
        # shape = "Unknown"
        
        if sides == 4:
            shape = "Red Square (Card)"
            found_red_card = True # 发现红牌！
            
        elif sides > 5:
            shape = "Red Ball"
            found_red_ball = True # 发现红球！

        # 在形状中心写上它的名字
        cv2.putText(frame_copy, shape, (cx - 40, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

    # ----------------- 5. 机器人动作决策大脑 -----------------
    # 逻辑要求：遇到红牌和人时，停止。遇到球时，跟踪。
    # 优先级：停止指令优先于跟踪指令 (安全第一)


    # 将最终指令打印在屏幕左上角
 
    # ----------------- 6. 显示与退出 -----------------
    # 可以同时显示掩膜图，方便你调试红色的阈值
    # cv2.imshow('Red Mask', mask) 
    cv2.imshow('Local Camera Vision', frame_copy)
    
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 【修改处 3】：释放摄像头资源
cap.release()
cv2.destroyAllWindows()