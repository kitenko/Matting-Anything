import os
import requests
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor
import zipfile


def download_file(url, dest_path):
    """
    Синхронная загрузка файла.
    :param url: Ссылка на файл.
    :param dest_path: Локальный путь для сохранения файла.
    """
    if os.path.exists(dest_path):
        print(f"Файл уже существует: {dest_path}")
        return

    print(f"Скачивание: {url}")
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))

    with open(dest_path, "wb") as file, tqdm(
        desc=f"Сохранение в {os.path.basename(dest_path)}",
        total=total_size,
        unit="B",
        unit_scale=True,
        unit_divisor=1024,
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            file.write(chunk)
            bar.update(len(chunk))


def download_all(urls, dest_dir, max_threads=4):
    """
    Загрузка всех файлов с использованием многопоточности.
    :param urls: Список URL для загрузки.
    :param dest_dir: Директория для сохранения файлов.
    :param max_threads: Максимальное количество потоков.
    """
    os.makedirs(dest_dir, exist_ok=True)

    with ThreadPoolExecutor(max_threads) as executor:
        for url in urls:
            dest_path = os.path.join(dest_dir, os.path.basename(url))
            executor.submit(download_file, url, dest_path)


def extract_zip(zip_path, extract_to):
    """
    Распаковка ZIP-архива.
    :param zip_path: Путь к ZIP-файлу.
    :param extract_to: Директория для распаковки.
    """
    print(f"Распаковка: {zip_path} -> {extract_to}")
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(extract_to)


def extract_all(zip_dir, extract_dir):
    """
    Распаковывает все ZIP-файлы в указанной директории.
    :param zip_dir: Директория, где хранятся ZIP-файлы.
    :param extract_dir: Директория для распаковки.
    """
    os.makedirs(extract_dir, exist_ok=True)
    for zip_file in os.listdir(zip_dir):
        if zip_file.endswith(".zip"):
            zip_path = os.path.join(zip_dir, zip_file)
            extract_zip(zip_path, extract_dir)


def prepare_coco_structure(data_dir):
    """
    Проверяет и организует стандартную структуру MS COCO.
    :param data_dir: Базовая директория для данных COCO.
    """
    print("Подготовка структуры MS COCO...")
    expected_dirs = ["train2017", "val2017", "annotations"]

    for sub_dir in expected_dirs:
        full_path = os.path.join(data_dir, sub_dir)
        if not os.path.exists(full_path):
            raise FileNotFoundError(f"Ожидаемая папка отсутствует: {full_path}")

    print("Структура MS COCO готова!")
    print(f"Тренировочные данные: {os.path.join(data_dir, 'train2017')}")
    print(f"Валидационные данные: {os.path.join(data_dir, 'val2017')}")
    print(f"Аннотации: {os.path.join(data_dir, 'annotations')}")


def main():
    # Директория для сохранения данных
    data_dir = "./coco_data"
    year = "2017"

    # Ссылки на файлы MS COCO
    base_url = "http://images.cocodataset.org"
    urls = [
        f"{base_url}/zips/train{year}.zip",
        f"{base_url}/zips/val{year}.zip",
        f"{base_url}/annotations/annotations_trainval{year}.zip",
    ]

    # Папка для загрузки ZIP-файлов
    zip_dir = os.path.join(data_dir, "zips")

    # Загрузка данных
    print("=== Загрузка MS COCO ===")
    download_all(urls, zip_dir, max_threads=4)

    # Распаковка данных
    print("=== Распаковка MS COCO ===")
    extract_all(zip_dir, data_dir)

    # Проверка и подготовка структуры
    prepare_coco_structure(data_dir)

    # Удаление ZIP-файлов
    print("=== Удаление ZIP-файлов ===")
    for file in os.listdir(zip_dir):
        os.remove(os.path.join(zip_dir, file))
    os.rmdir(zip_dir)

    print("=== MS COCO успешно загружен и подготовлен! ===")


if __name__ == "__main__":
    main()
