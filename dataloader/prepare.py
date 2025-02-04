import os
import cv2
import numpy as np
from collections import defaultdict
from tqdm import tqdm  # pip install tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

def _merge_and_save(args):
    """
    Функция-рабочий, которая объединяет маски (операцией np.maximum) 
    и сохраняет результат.
    Принимает кортеж (base_name, paths, save_dir).
    """
    base_name, paths, save_dir = args
    alpha_merged = None

    for p in paths:
        mask = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if mask is None:
            # Можно залогировать ошибку
            continue
        mask_float = mask.astype(np.float32) / 255.0
        alpha_merged = mask_float if alpha_merged is None else np.maximum(alpha_merged, mask_float)

    # Если не удалось считать ни одной маски, пропускаем
    if alpha_merged is None:
        return False

    merged_uint8 = (alpha_merged * 255).clip(0, 255).astype(np.uint8)
    out_path = os.path.join(save_dir, f"{base_name}.png")
    cv2.imwrite(out_path, merged_uint8)
    return True


def combine_masks(mask_dir, save_dir):
    """
    Сканирует mask_dir, группирует маски по базовым именам (без суффиксов _1, _2, ...),
    объединяет каждую группу в одну маску (np.maximum) и сохраняет в save_dir.
    Использует ProcessPoolExecutor(max_workers=6) для параллельной обработки.
    """
    os.makedirs(save_dir, exist_ok=True)

    if not os.path.isdir(mask_dir):
        raise FileNotFoundError(f"Директория не найдена: {mask_dir}")

    # Фильтр: только изображения
    def is_image(fname):
        IMG_EXTENSIONS = ('.jpg', '.jpeg', '.png', '.bmp', '.tif', '.tiff')
        return fname.lower().endswith(IMG_EXTENSIONS)

    mask_files = sorted([f for f in os.listdir(mask_dir) if is_image(f)])
    if not mask_files:
        print("В директории масок нет изображений.")
        return

    # Функция для срезания суффикса _1, _2 и т.д.
    def get_base_key(filename):
        name, _ = os.path.splitext(filename)
        if '_' in name:
            parts = name.rsplit('_', 1)
            if parts[1].isdigit():
                return parts[0]
        return name

    # Группируем по base_key
    mask_dict = defaultdict(list)
    for fname in mask_files:
        full_path = os.path.join(mask_dir, fname)
        base = get_base_key(fname)
        mask_dict[base].append(full_path)

    print(f"Найдено {len(mask_files)} файлов масок, сгруппированных в {len(mask_dict)} блоков.")

    # Создаём задачи для параллельной обработки
    tasks = [(base_name, paths, save_dir) for base_name, paths in mask_dict.items()]

    # Запускаем пул процессов (6) и отслеживаем прогресс
    with ProcessPoolExecutor(max_workers=6) as executor:
        # submit -> возвращает Future
        future_to_base = {executor.submit(_merge_and_save, t): t[0] for t in tasks}

        # tqdm по количеству блоков
        for future in tqdm(as_completed(future_to_base), 
                           total=len(tasks), 
                           desc="Обработка групп масок"):
            base_name = future_to_base[future]
            # Если нужно - отлавливаем исключения:
            try:
                result = future.result()
                # result == True если успешно сохранено, иначе False
            except Exception as e:
                print(f"\nОшибка при обработке {base_name}: {e}")

    print(f"\nГотово! Объединённые маски сохранены в: {save_dir}")


if __name__ == '__main__':
    mask_dir = '/app/datasets/RefMatte/train/mask'
    save_dir = '/app/datasets/RefMatte/train/prepare_mask'
    combine_masks(mask_dir, save_dir)
