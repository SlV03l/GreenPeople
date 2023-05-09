import cv2
import numpy as np
import bluetooth
import time
import threading

socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
socket.connect(('98:D3:71:F6:5D:5D', 1))
IP = "http://192.168.251.192:8080/video?.mjpeg%22"

OBJECT_WIDTH = 30.7 # Ширина объекта в сантиметрах
OBJECT_HEIGHT = 22 # Высота объекта в сантиметрах
IMAGE_WIDTH = 1920 # Ширина изображения в пикселях
IMAGE_HEIGHT = 1080 # Высота изображения в пикселях
SENSOR_WIDTH = 5.6 # Ширина сенсора камеры в мм
FOCAL_LENGTH = 26 # Фокусное расстояние в мм

DEL = 0.08
MinDist = 100
DST = '200'
Dog = 'OFF'
START = False
DogMode = 'None'
colors = ['None', 'blue', 'green', 'red', 'purple', 'yellow', 'lilac']
index = 0
last_time = time.time()
BLIND_SQUARE = 80

color_ranges = {'yellow': [np.array([10, 50, 50]), np.array([35, 255, 255])],
                'blue': [np.array([90, 100, 50]), np.array([128, 255, 255])],
                'red': [np.array([100, 100, 100]), np.array([10, 255, 255]),
                        np.array([170, 100, 80]), np.array([180, 255, 255])],
                'purple': [np.array([120, 50, 50]), np.array([150, 255, 255])],
                'green': [np.array([50, 50, 50]), np.array([70, 255, 255])],
                'lilac': [np.array([140, 50, 50]), np.array([170, 255, 255])]
                }

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)
cap.open(IP)

def toggle_bot1(dx):
    if dx > 0:
        socket.send('3'+str(dx)+" ")
        print('Вправо', dx)
    elif dx < 0:
        socket.send('2'+str(dx)+" ")
        print('Влево', dx)

def toggle_bot2(dy):
    if dy > 0:
        socket.send('5')
        print('Вниз')
    elif dy < 0:
        socket.send('6')
        print('Вверх')

def toggle_bot3(distance):
    global MinDist
    if distance > MinDist:
        socket.send('1')
        print(f"{distance} -> Вперед")

def toggle_start():
    global START
    global Dog
    if not START:
        START = True
        Dog = 'ON'
    else:
        START = False
        Dog = 'OFF'

def update_color(dir):
    global index, DogMode
    if dir == 'right':
        index += 1
        if index == len(colors):
            index = 0
    elif dir == 'left':
        index -= 1
        if index == -1:
            index = len(colors) - 1
    DogMode = colors[index]


def calculate_distance(contour):
    # вычисление расстояния до объекта
    x, y, w, h = cv2.boundingRect(contour)
    object_width_in_pixels = max(w, h)
    object_height_in_pixels = min(w, h)

    sensor_width_in_pixels = IMAGE_WIDTH / SENSOR_WIDTH
    focal_length_in_pixels = FOCAL_LENGTH * sensor_width_in_pixels / SENSOR_WIDTH
    distance_width = (OBJECT_WIDTH * focal_length_in_pixels) / object_width_in_pixels
    distance_height = (OBJECT_HEIGHT * focal_length_in_pixels) / object_height_in_pixels
    distance = int(max(distance_width, distance_height))

    return distance

def process_color_range(hsv, color_range, color_name, frame):
    mask = cv2.inRange(hsv, color_range[0], color_range[1])
    find_and_draw_contours(mask, color_name, frame)

def draw_grid(frames):
    cell_size_x = frames.shape[1] // 3
    cell_size_y = frames.shape[0] // 3
    for i in range(cell_size_x, frames.shape[1], cell_size_x):
        cv2.line(frames, (i, 0), (i, frames.shape[0]), (255, 255, 255), 2)
    for j in range(cell_size_y, frames.shape[0], cell_size_y):
        cv2.line(frames, (0, j), (frames.shape[1], j), (255, 255, 255), 2)
    center_x = frames.shape[1] // 2
    center_y = frames.shape[0] // 2
    cross_size = min(80, 80)
    cv2.line(frames, (center_x - cross_size, center_y), (center_x + cross_size, center_y), (194, 39, 178), 6)
    cv2.line(frames, (center_x, center_y - cross_size), (center_x, center_y + cross_size), (194, 39, 178), 6)
    cv2.putText(frames, 'Color: ' + DogMode, (center_x - 945, center_y + 460), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    cv2.putText(frames, 'DogMode: ' + Dog, (center_x - 945, center_y + 520), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    return frames

def controlling(contour, cX, cY):
    # Получаем размеры кадра
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    # Вычисляем координаты центра кадра
    center_x = int(frame_width / 2)
    center_y = int(frame_height / 2)

    global last_time
    if time.time() - last_time > DEL:
        last_time = time.time()
        distance_to_center_x = abs(center_x - cX)
        distance_to_center_y = abs(center_y - cY)
        # Вычисляем разницу между центром кадра и центром масс объекта
        dx = cX - center_x
        dy = cY - center_y

        dist = calculate_distance(contour)

        if distance_to_center_y > BLIND_SQUARE:
            toggle_bot2(dy)

        if distance_to_center_x > BLIND_SQUARE:
            toggle_bot1(dx)

        toggle_bot3(dist)

def find_and_draw_contours(mask, color_name, frame):
    ret, thresh = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for contour in contours:
        area = cv2.contourArea(contour)
        if area > 5000:
            contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(contour)
            aspect_ratio = float(w) / h
            if aspect_ratio > 0.5:
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                cv2.drawContours(frame, [approx], 0, (0, 255, 0), 3)
                cv2.putText(frame, color_name, (approx[0][0][0], approx[0][0][1]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

                M = cv2.moments(contour)
                if M["m00"] != 0:
                    cX = int(M["m10"] / M["m00"])
                    cY = int(M["m01"] / M["m00"])
                    cv2.circle(frame, (cX, cY), 20, (0, 255, 0), -1)
                    text = "({}, {})".format(cX, cY)
                    cv2.putText(frame, text, (cX + 25, cY), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
                    if START:
                        threading.Thread(target=controlling, args=(contour, cX, cY)).start()


while True:
    ret, frame = cap.read()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    draw_grid(frame)

    for color_name, ranges in color_ranges.items():
        mask = cv2.inRange(hsv, ranges[0], ranges[1])
        if len(ranges) > 2:
            mask = cv2.bitwise_or(cv2.inRange(hsv, ranges[2], ranges[3]), mask)
        if color_name == DogMode:
            threading.Thread(target=find_and_draw_contours, args=(mask, color_name, frame)).start()

    resized_frame = cv2.resize(frame, (1280, 720))
    cv2.imshow('AgroVision 1.1.2', resized_frame)

    def exit_and_close_socket():
        socket.close()
        exit()

    key = cv2.waitKey(1)
    key_dict = {
        27: lambda: exit_and_close_socket(),
        ord('='): lambda: socket.send('5'),
        ord('-'): lambda: socket.send('6'),
        ord('a'): lambda: socket.send('20 '),
        ord('d'): lambda: socket.send('30 '),
        ord('w'): lambda: socket.send('1'),
        ord('s'): lambda: socket.send('4r'),
        ord('r'): lambda: toggle_start(),
        ord('.'): lambda: update_color('right'),
        ord(','): lambda: update_color('left'),
    }
    key_dict.get(key, lambda: None)()

cap.release()
cv2.destroyAllWindows()