import os
from roboflow import Roboflow
import subprocess
import ultralytics


API_KEY = ""  # TODO: Add your Roboflow API key here
WORKSPACE = "yolo-j3jjb"
PROJECT = "merged-fruitninja"
BASE_MODEL = "yolov8"


def change_dir(folder):
    HOME = os.getcwd()
    print(f'CURRENT PATH: {HOME}')
    dataset_path = f'{HOME}/{folder}'

    if not os.path.exists(dataset_path):
        os.mkdir(dataset_path)

    os.chdir(dataset_path)


def download_data():
    rf = Roboflow(api_key=API_KEY)
    project = rf.workspace(WORKSPACE).project(PROJECT)
    dataset = project.version(4).download(BASE_MODEL)

    return dataset.location


def train_model(data_path, epochs=30, base_model="yolov8n"):
    command = [
        "yolo",
        "task=detect",
        "mode=train",
        f"model={base_model}.pt",
        f"data={data_path}/data.yaml",
        f"epochs={epochs}",
        "imgsz=800",
        "plots=True",
        "batch=10"
    ]

    process = subprocess.Popen(command, stdout=subprocess.PIPE, universal_newlines=True)

    if process.stdout:
        for line in process.stdout:
            print(line.strip())
    else:
        print('Training model result is None...')

    print('End training model...')


if __name__ == '__main__':
    ultralytics.checks()

    change_dir('datasets')
    data_folder = download_data()

    print(f'DATA YAML folder: {data_folder}')
    print('Start training model...')

    train_model(data_folder, epochs=1)

