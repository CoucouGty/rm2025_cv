import time
import cv2
import numpy as np
import math
video = cv2.VideoCapture(r'red.avi')
previous_zhongpoint = None
previous_velocity = None
target_color = "blue"  # 可切换

def distance(p1, p2):
    return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

def extract_color_channel(img, target_color):
    blue, green, red = cv2.split(img)
    if target_color == "blue":
        return blue
    elif target_color == "red":
        return red
    elif target_color == "green":
        return green
    else:
        raise ValueError("Unsupported color selection!")

while True:
    ret, img = video.read()
    if not ret:
        break
    start_time = time.time()
    color_channel = extract_color_channel(img, target_color)

    ret2, binary = cv2.threshold(color_channel, 220, 255, 0)
    Gaussian = cv2.GaussianBlur(binary, (5, 5), 0)  # 进行高斯滤波

    draw_img = Gaussian.copy()
    whole_h, whole_w = binary.shape[:2]

    contours, hierarchy = cv2.findContours(image=draw_img, mode=cv2.RETR_TREE, method=cv2.CHAIN_APPROX_NONE)
    contours = list(contours)
    contours.sort(key=lambda c: cv2.contourArea(c), reverse=True)

    width_array = []
    height_array = []
    point_array = []

    for cont in contours[:5]:
        x, y, w, h = cv2.boundingRect(cont)
        try:
            if h / w >= 2 and h / whole_h > 0.1 and h > w:
                width_array.append(w)
                height_array.append(h)
                point_array.append([x, y])
        except:
            continue

    point_near = [0, 0]
    min_value = 10000
    for i in range(len(width_array) - 1):
        for j in range(i + 1, len(width_array)):
            value = abs(width_array[i] * height_array[i] - width_array[j] * height_array[j])
            if value < min_value:
                min_value = value
                point_near[0] = i
                point_near[1] = j

    try:
        rectangle1 = point_array[point_near[0]]
        rectangle2 = point_array[point_near[1]]

        point1 = [rectangle1[0] + width_array[point_near[0]] / 2, rectangle1[1]]
        point2 = [rectangle1[0] + width_array[point_near[0]] / 2, rectangle1[1] + height_array[point_near[0]]]
        point3 = [rectangle2[0] + width_array[point_near[1]] / 2, rectangle2[1]]
        point4 = [rectangle2[0] + width_array[point_near[1]] / 2, rectangle2[1] + height_array[point_near[1]]]

        zhongpoint = [(point1[0] + point4[0]) / 2, (point1[1] + point4[1]) / 2]
        print("当前帧中点:", zhongpoint)

        if previous_zhongpoint is not None:
            current_velocity = [(zhongpoint[0] - previous_zhongpoint[0]), (zhongpoint[1] - previous_zhongpoint[1])]
            print("当前速度:", current_velocity)

            if previous_velocity is not None:
                acceleration = [(current_velocity[0] - previous_velocity[0]),
                                (current_velocity[1] - previous_velocity[1])]
                print("加速度:", acceleration)

                predicted_zhongpoint = [
                    zhongpoint[0] + current_velocity[0] + 0.5 * acceleration[0],
                    zhongpoint[1] + current_velocity[1] + 0.5 * acceleration[1]
                ]
                print("预测下一帧中点:", predicted_zhongpoint)

            previous_velocity = current_velocity

        previous_zhongpoint = zhongpoint

        cv2.circle(img, (int(zhongpoint[0]), int(zhongpoint[1])), 5, (0, 0, 255), -1)  # 用红色圆圈标记中点

        x = np.array([point1, point2, point3, point4], np.int32)
        box = x.reshape((-1, 1, 2)).astype(np.int32)
        cv2.polylines(img, [box], True, (0, 255, 0), 2)

    except Exception as e:
        print(f"Error: {e}")
        continue

    time.sleep(0.01)
    cv2.imshow('name', img)
    end_time = time.time()
    print(f'处理一帧耗时: {(end_time - start_time) * 1000:.2f} ms')
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video.release()
cv2.destroyAllWindows()
