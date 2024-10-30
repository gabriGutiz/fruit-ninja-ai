import cv2
from mss import mss
import numpy as np
from PIL import Image
from pynput.mouse import Button, Controller
import time
import torch
from ultralytics import YOLO


BOMB_CLASS = "Bomb"
FRUIT_CLASS = "Fruit"
CONFIDENCE_THRESHOLD = 0.8


sct = mss()
mouse = Controller()


def get_screen_box(window_box):
    sct_img = sct.grab(window_box)

    sct_img = Image.frombytes(
        'RGB',
        (sct_img.width, sct_img.height),
        sct_img.rgb,
    )

    return cv2.cvtColor(np.array(sct_img), cv2.COLOR_RGB2BGR)


def detect_fruits_and_bombs(img, model, device):
    results = model(source=img, device=device, verbose=False, iou=0.25, conf=CONFIDENCE_THRESHOLD, int8=True, imgsz=(640,640))

    frame = results[0].plot()

    detected_fruits, detected_bombs = [], []

    result = results[0]

    for box, cls in zip(result.boxes.xyxy.tolist(), result.boxes.cls.tolist()):
        if result.names[int(cls)] == BOMB_CLASS:
            detected_bombs.append(box)
        elif result.names[int(cls)] == FRUIT_CLASS:
            detected_fruits.append(box)

    return (frame, detected_fruits, detected_bombs)


def sleep(duration):
    end_time = time.time() + duration
    while time.time() < end_time:
        pass


def cut(cuts, padding_x, padding_y):
    cuts_done = []

    steps = 30

    for cut in cuts:
        mouse.release(Button.left)

        x1, y1, x2, y2 = cut

        x1 = int(x1)
        y1 = int(y1)
        x2 = int(x2)
        y2 = int(y2)

        p_x = int((x2 - x1) * 0.35)
        p_y = int((y2 - y1) * 0.35)

        pos1 = (x1 + p_x, y1 + p_y)
        pos2 = (x2 - p_x, y2 - p_y)
        cuts_done.append((pos1, pos2))

        mouse.position = (x1 + p_x + padding_x, y1 + p_y + padding_y)
        mouse.press(Button.left)

        diff_x = pos2[0] - pos1[0]
        diff_y = pos2[1] - pos1[1]

        for p in range(steps):
            x = pos1[0] + int(diff_x * p / steps)
            y = pos1[1] + int(diff_y * p / steps)

            mouse.position = (x + padding_x, y + padding_y)
            sleep(0.0001)

        mouse.release(Button.left)

        sleep(0.01)

    return cuts_done


def run_fruit_ninja_ai(model_path: str, game_box: dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = YOLO(model_path)

    w, h = game_box['width']/2, game_box['height']/2

    while True:
        game_screen = get_screen_box(game_box)

        game_screen, fruits, bombs = detect_fruits_and_bombs(game_screen, model, device)

        fruits = [fruit for fruit in fruits if fruit[1] < 0.5*game_box['height']]

        cuts = cut(fruits, game_box['left'], game_box['top'])

        for c in cuts:
            game_screen = cv2.line(game_screen, c[0], c[1], (0, 255, 0), 2)

        game_screen = cv2.resize(game_screen, (int(w), int(h)))

        win_name = 'vision'
        cv2.imshow(win_name, game_screen)
        cv2.moveWindow(win_name, 1920, 1080)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break


if __name__ == '__main__':
    model_path = 'best.pt'
    
    game_box = {'top': 130, 'left': 365, 'width': 1040, 'height': 590}

    run_fruit_ninja_ai(model_path, game_box)

