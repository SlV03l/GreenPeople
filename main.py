import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap, QIcon, QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QLabel, QMessageBox, QDialog, QVBoxLayout, \
    QLineEdit, QGridLayout, QScrollArea, QWidget, QCheckBox
import cv2
import numpy as np
import bluetooth
import time
import threading
import torch

IP = "http://192.168.172.51:8080/video?.mjpeg%22"

socket = 0
socket_open = False

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
cap.set(cv2.CAP_PROP_FPS, 30)

model = torch.hub.load('ultralytics/yolov5', 'custom', path='med-P6-gen33.pt', force_reload=True)

camera_update = True

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

Face = False
Face_Cont = 'OFF'
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

Sign = False
Sign_Cont = 'OFF'

def face_search():
    global Face
    global Face_Cont
    if not Face:
        Face = True
        Face_Cont = 'ON'
    else:
        Face = False
        Face_Cont = 'OFF'

def sign_search():
    global Sign
    global Sign_Cont
    if not Sign:
        Sign = True
        Sign_Cont = 'ON'
    else:
        Sign = False
        Sign_Cont = 'OFF'

def exit_and_close_socket():
    global socket_open
    if socket_open:
        socket.close()
    exit()
def toggle_bot1(dx):
    global socket_open
    if socket_open:
        if dx > 0:
            socket.send('3'+str(dx)+" ")
            print('Вправо', dx)
        elif dx < 0:
            socket.send('2'+str(dx)+" ")
            print('Влево', dx)

def toggle_bot2(dy):
    global socket_open
    if socket_open:
        if dy > 0:
            socket.send('5')
            print('Вниз')
        elif dy < 0:
            socket.send('6')
            print('Вверх')

def toggle_bot3(distance):
    global MinDist
    global socket_open
    if socket_open:
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
    global camera_update
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
    if camera_update:
        cv2.putText(frames, 'Road sign: ' + Sign_Cont, (center_x - 945, center_y - 500), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frames, 'Face: ' + Face_Cont, (center_x - 945, center_y + 400), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frames, 'Color: ' + DogMode, (center_x - 945, center_y + 460), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frames, 'DogMode: ' + Dog, (center_x - 945, center_y + 520), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
    else:
        cv2.putText(frames, 'Road sign: ' + Sign_Cont, (center_x - 610, center_y - 300), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frames, 'Face: ' + Face_Cont, (center_x - 610, center_y + 200), cv2.FONT_HERSHEY_TRIPLEX, 1.5,(255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frames, 'Color: ' + DogMode, (center_x - 610, center_y + 260), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frames, 'DogMode: ' + Dog, (center_x - 610, center_y + 320), cv2.FONT_HERSHEY_TRIPLEX, 1.5, (255, 255, 255), 3, cv2.LINE_AA)
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

def draw_sign_boxes(frame, detections):
    for box in detections.xyxy[0]:
        x1, y1, x2, y2 = box[:4].int().tolist()
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
        label = f"{detections.names[int(box[5])]} {box[4]:.2f}"
        cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        class_name = detections.names[int(box[5])]

        if class_name == '3_24':
            print('Едь медленнее, а то угодишь в канаву')

def find_and_draw_contours(mask, color_name, frame):
    if Face:
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        contours = []
        for (x, y, w, h) in faces:
            contours.append(np.array([[x, y], [x + w, y], [x + w, y + h], [x, y + h]], dtype=np.int32))
    else:
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

class SettingsWindow(QDialog):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("Настройки")
        self.resize(600, 300)

        background_image_path = 'SettingAg.png'

        # Установка фонового изображения
        background_image = QPixmap(background_image_path)
        background_label = QLabel(self)
        background_label.setPixmap(background_image)
        background_label.setScaledContents(True)
        background_label.setGeometry(0, 0, self.width(), self.height())

        layout = QGridLayout()

        self.font = QFont("times new roman", 14, QFont.Bold)  # Сделать font атрибутом класса
        # Лейбл IP
        ip_label = QLabel("CAMERA IP:")
        ip_label.setFont(self.font)
        layout.addWidget(ip_label, 0, 0)

        # Поле ввода IP
        global IP
        self.ip_input = QLineEdit()
        self.ip_input.setText(IP[7:24])
        self.ip_input.setStyleSheet("border-radius: 5px; background-color: rgba(255, 255, 255, 0.6); padding: 8px;")
        layout.addWidget(self.ip_input, 1, 0)

        # Лейбл выбора камер
        camera_selection_label = QLabel("Устройство видеопотока: ")
        camera_selection_label.setFont(self.font)
        layout.addWidget(camera_selection_label, 7, 0, alignment=Qt.AlignHCenter)

        # Лейбл Подключить bluetooth:
        auto_connection_label = QLabel("Подключить bluetooth: ")
        auto_connection_label.setFont(self.font)
        layout.addWidget(auto_connection_label, 2, 0, alignment=Qt.AlignHCenter)

        # Чекбокс auto_connection
        self.auto_connection_checkbox = QCheckBox()
        layout.addWidget(self.auto_connection_checkbox, 2, 1, alignment=Qt.AlignHCenter)
        self.auto_connection_checkbox.stateChanged.connect(self.auto_connection_changed)

        # Лейбл Color
        color_label = QLabel("Color:")
        color_label.setFont(self.font)
        layout.addWidget(color_label, 0, 1, alignment=Qt.AlignHCenter)

        # Поле ввода Color
        self.color_input = QLineEdit()
        self.color_input.setStyleSheet("border-radius: 5px; background-color: rgba(255, 255, 255, 0.6); padding: 8px;")
        layout.addWidget(self.color_input, 1, 1)

        # Лейбл Color_Range
        color_range_label = QLabel("Диапазон цвета:")
        color_range_label.setFont(self.font)
        layout.addWidget(color_range_label, 0, 2, alignment=Qt.AlignHCenter)

        # Поле ввода lower_range
        self.lower_range_input = QLineEdit()
        self.lower_range_input.setStyleSheet("border-radius: 5px; background-color: rgba(255, 255, 255, 0.6); padding: 8px;")
        layout.addWidget(self.lower_range_input, 1, 2)

        # Поле ввода upper_range
        self.upper_range_input = QLineEdit()
        self.upper_range_input.setStyleSheet("border-radius: 5px; background-color: rgba(255, 255, 255, 0.6); padding: 8px;")
        layout.addWidget(self.upper_range_input, 2, 2)

        button_transparency = 0.1
        # Кнопка добавить
        add_button = QPushButton("Добавить")
        add_button.setStyleSheet(
            f"QPushButton {{ background-color: rgba(0, 0, 0, {button_transparency}); font-size: 24px; font-family: Comic Sans MS; border-radius: 15px; }}")
        layout.addWidget(add_button, 3, 0)

        # Кнопка удалить
        remove_button = QPushButton("Удалить")
        remove_button.setStyleSheet(
            f"QPushButton {{ background-color: rgba(0, 0, 0, {button_transparency}); font-size: 24px; font-family: Comic Sans MS; border-radius: 15px; }}")
        layout.addWidget(remove_button, 4, 0)

        # Кнопка изменить
        update_button = QPushButton("Изменить")
        update_button.setStyleSheet(
            f"QPushButton {{ background-color: rgba(0, 0, 0, {button_transparency}); font-size: 24px; font-family: Comic Sans MS; border-radius: 15px; }}")
        layout.addWidget(update_button, 5, 0)

        # Кнопка сохранить
        save_button = QPushButton("Сохранить IP")
        save_button.setStyleSheet(
            f"QPushButton {{ background-color: rgba(125, 209, 156, {0}); font-size: 20px; font-family: Comic Sans MS; border-radius: 20px; }}")
        layout.addWidget(save_button, 6, 0)

        # Кнопка выбора камеры
        self.camera_button = QPushButton("IP Webcam")
        self.camera_button.setStyleSheet(
            f"QPushButton {{ background-color: rgba(0, 0, 0, {button_transparency}); font-size: 24px; font-family: Comic Sans MS; border-radius: 15px; }}")
        self.camera_button.clicked.connect(self.toggle_camera)
        layout.addWidget(self.camera_button, 8, 0)


        # Создание виджета для поля со скроллом
        self.scroll_widget = QWidget()
        self.scroll_layout = QVBoxLayout(self.scroll_widget)
        self.scroll_layout.setSpacing(10)

        # Глобальная переменная color_ranges
        global color_ranges
        global colors
        for i, (key, value) in enumerate(color_ranges.items()):
            label = QLabel(f"{key}: {value}")
            label.setFont(self.font)
            label.setStyleSheet("font-size: 12px;")
            self.scroll_layout.addWidget(label)

        # Создание виджета скролла и добавление поля со скроллом
        self.scroll_area = QScrollArea()
        self.scroll_area.setWidgetResizable(True)
        self.scroll_area.setStyleSheet("background-color: rgba(255, 255, 255, 0.3);")
        self.scroll_area.setWidget(self.scroll_widget)
        layout.addWidget(self.scroll_area, 3, 1, -1, 2)

        # Обработчик для кнопок
        remove_button.clicked.connect(self.remove_color_range)
        add_button.clicked.connect(self.add_color)
        update_button.clicked.connect(self.update_color_range)
        save_button.clicked.connect(self.save_ip)

        self.setLayout(layout)

    def toggle_camera(self):
        global camera_update
        if self.camera_button.text() == "IP Webcam":
            self.camera_button.setText("PC Webcam")
            camera_update = False
        else:
            self.camera_button.setText("IP Webcam")
            camera_update = True

    def add_color(self):
        color_input_text = self.color_input.text()
        lower_range_text = self.lower_range_input.text()
        upper_range_text = self.upper_range_input.text()

        if not color_input_text or not lower_range_text or not upper_range_text:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, заполните все поля.")
        else:
            lower_range_values = [int(x.strip()) for x in lower_range_text.split(',')]
            upper_range_values = [int(x.strip()) for x in upper_range_text.split(',')]

            if len(lower_range_values) == 3 or len(lower_range_values) == 2 and len(upper_range_values) == 3 or len(
                    upper_range_values) == 2:
                lower_range_np = np.array(lower_range_values)
                upper_range_np = np.array(upper_range_values)

                color_ranges[color_input_text] = [lower_range_np, upper_range_np]
                colors.append(color_input_text)
                self.color_input.clear()
                self.lower_range_input.clear()
                self.upper_range_input.clear()
                QMessageBox.information(self, "Успех", f"Цвет {color_input_text} успешно добавлен.")
                self.update_color_range_labels()
            else:
                QMessageBox.warning(self, 'Ошибка', 'Некорректные значения диапазона цвета.')

    def auto_connection_changed(self, state):
        # Обработчик изменения состояния чекбокса auto_connection
        global socket
        global socket_open
        #
        if state == Qt.Checked:
            print("Чекбокс auto_connection отмечен")
            socket_open = True
            QMessageBox.warning(self, "Подключение", "Устройство подключено по Bluetooth!")
            try:
                socket = bluetooth.BluetoothSocket(bluetooth.RFCOMM)
                QMessageBox.warning(self, "Успешно", "Устройство подключено по Bluetooth! ")
            except bluetooth.btcommon.BluetoothError as e:
                QMessageBox.warning(self, "Ошибка", "Ошибка при создании сокета Bluetooth: " + str(e))
            else:
                try:
                    socket.connect(('98:D3:71:F6:5D:5D', 1))
                except bluetooth.btcommon.BluetoothError as e:
                    QMessageBox.warning(self, "Ошибка", "Ошибка подключения к устройству Bluetooth: " + str(e))
        else:
            if socket_open:
                QMessageBox.warning(self, "Отключение", "Устройство отсоединено!")
                print("Чекбокс auto_connection снят")
                socket.close()
                socket_open = False

    def remove_color_range(self):
        color_input_text = self.color_input.text()
        if color_input_text in color_ranges:
            del color_ranges[color_input_text]
            colors.remove(color_input_text)
            self.color_input.clear()
            QMessageBox.information(self, "Успех", f"Цвет {color_input_text} успешно удален.")
            self.update_color_range_labels()
        else:
            QMessageBox.warning(self, "Ошибка", f"Цвет {color_input_text} не найден.")

    def update_color_range_labels(self):
        # Очистка виджета со скроллом
        scroll_widget = self.scroll_area.widget()
        scroll_layout = scroll_widget.layout()
        while scroll_layout.count():
            item = scroll_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()

        # Перерисовка виджета со скроллом
        for i, (key, value) in enumerate(color_ranges.items()):
            label = QLabel(f"{key}: {value}")
            label.setFont(self.font)
            label.setStyleSheet("font-size: 12px;")
            scroll_layout.addWidget(label)

    def update_color_range(self):
        color_input_text = self.color_input.text()
        lower_range_text = self.lower_range_input.text()
        upper_range_text = self.upper_range_input.text()

        if not color_input_text or not lower_range_text or not upper_range_text:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, заполните все поля.")
        else:
            lower_range_values = [int(x.strip()) for x in lower_range_text.split(',')]
            upper_range_values = [int(x.strip()) for x in upper_range_text.split(',')]

            if len(lower_range_values) == 3 or len(lower_range_values) == 2 and len(upper_range_values) == 3 or len(
                    upper_range_values) == 2:
                lower_range_np = np.array(lower_range_values)
                upper_range_np = np.array(upper_range_values)

                color_ranges[color_input_text] = [lower_range_np, upper_range_np]
                self.color_input.clear()
                self.lower_range_input.clear()
                self.upper_range_input.clear()
                QMessageBox.information(self, "Успех", f"Цвет {color_input_text} успешно изменен.")
                self.update_color_range_labels()
            else:
                QMessageBox.warning(self, 'Ошибка', 'Некорректные значения диапазона цвета.')

    def save_ip(self):
        ip_text = self.ip_input.text()

        if not ip_text:
            QMessageBox.warning(self, "Ошибка", "Пожалуйста, введите IP-адрес.")
        else:
            global IP
            IP = f"http://{ip_text}/video?.mjpeg%22"
            QMessageBox.information(self, "Успех", "IP успешно сохранен.")
class MainWindow(QMainWindow):
    def __init__(self, background_image_path):
        super().__init__()

        icon = QIcon("AgroVision.png")
        self.setWindowIcon(icon)

        # Установка заднего фона
        background_image = QPixmap(background_image_path)
        background_label = QLabel(self)
        background_label.setPixmap(background_image)
        background_label.setScaledContents(True)

        # Получение размеров фона
        background_size = background_image.size()

        # Установка размеров окна
        self.setFixedSize(background_size)

        # Создание кнопок
        start_button = QPushButton('Начать', self)
        settings_button = QPushButton('Настройки', self)
        exit_button = QPushButton('Выход', self)

        # Расположение кнопок
        button_margin = 20 * 5
        button_width = 70
        button_height = 20
        start_button.move((background_size.width() - button_width * 3) // 2,
                          (background_size.height() - button_height) // 2 - button_margin)
        settings_button.move((background_size.width() - button_width * 3) // 2,
                             (background_size.height() - button_height) // 2)
        exit_button.move((background_size.width() - button_width * 3) // 2,
                         (background_size.height() - button_height) // 2 + button_margin)

        button_width = 120 * 2
        button_height = 40 * 2
        start_button.resize(button_width, button_height)
        settings_button.resize(button_width, button_height)
        exit_button.resize(button_width, button_height)

        button_transparency = 0.6
        start_button.setStyleSheet(
            f"QPushButton {{ background-color: rgba(255, 255, 255, {button_transparency}); font-size: 24px; font-family: Comic Sans MS; border-radius: 20px; }}"
        )
        settings_button.setStyleSheet(
            f"QPushButton {{ background-color: rgba(255, 255, 255, {button_transparency}); font-size: 24px; font-family: Comic Sans MS; border-radius: 20px; }}"
        )
        exit_button.setStyleSheet(
            f"QPushButton {{ background-color: rgba(255, 255, 255, {button_transparency}); font-size: 24px; font-family: Comic Sans MS; border-radius: 20px; }}"
        )

        # Установка обработчиков для кнопок
        start_button.clicked.connect(self.start_button_clicked)
        settings_button.clicked.connect(self.settings_button_clicked)
        exit_button.clicked.connect(self.exit_button_clicked)

        # Установка основного виджета
        self.setCentralWidget(background_label)

    def start_button_clicked(self):
        if camera_update:
            cap.open(IP)

        self.close()
        while True:
            ret, frame = cap.read()

            if Sign:
                res = model(frame)
                threading.Thread(target=draw_sign_boxes, args=(frame, res)).start()
            elif Face:
                mask = 1
                color_name = "face"
                find_and_draw_contours(mask, color_name, frame)
            else:
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
                for color_name, ranges in color_ranges.items():
                    mask = cv2.inRange(hsv, ranges[0], ranges[1])
                    if len(ranges) > 2:
                        mask = cv2.bitwise_or(cv2.inRange(hsv, ranges[2], ranges[3]), mask)
                    if color_name == DogMode:
                        threading.Thread(target=find_and_draw_contours, args=(mask, color_name, frame)).start()

            draw_grid(frame)
            resized_frame = cv2.resize(frame, (1280, 720))
            cv2.imshow('AgroVision 1.1.2', resized_frame)

            key = cv2.waitKey(1)
            key_dict = {
                27: lambda: exit_and_close_socket(),
                ord('='): lambda: socket.send('5'),
                ord('-'): lambda: socket.send('6'),
                ord('a'): lambda: socket.send('20 '),
                ord('d'): lambda: socket.send('30 '),
                ord('w'): lambda: socket.send('1'),
                ord('s'): lambda: socket.send('4'),
                ord('r'): lambda: toggle_start(),
                ord('.'): lambda: update_color('right'),
                ord(','): lambda: update_color('left'),
                ord('f'): lambda: face_search(),
                ord('q'): lambda: sign_search(),
            }
            key_dict.get(key, lambda: None)()

        cap.release()
        cv2.destroyAllWindows()

    def settings_button_clicked(self):
        settings_window = SettingsWindow()
        settings_window.exec_()

    def exit_button_clicked(self):
        reply = QMessageBox.question(self, 'Подтверждение выхода',
                                     'Вы действительно хотите выйти?',
                                     QMessageBox.Yes | QMessageBox.No,
                                     QMessageBox.No)

        if reply == QMessageBox.Yes:
            # Выбрано "Да" - закрыть программу
            self.close()


if __name__ == '__main__':
    app = QApplication(sys.argv)

    app_icon = QIcon('AgroVision.png')
    app.setWindowIcon(app_icon)

    background_image_path = 'AgroV.png'

    window = MainWindow(background_image_path)
    window.setWindowTitle("АгроЗрение - Сливницин РПС-21")
    window.show()

    sys.exit(app.exec_())