import os
import argparse
from pathlib import Path

def count_files(directory, extensions, recursive=False, list_files=False):
    """
    Подсчитывает количество файлов с заданными расширениями в директории.

    :param directory: Путь к директории.
    :param extensions: Список расширений файлов (без точки, например, ['png', 'jpg']).
    :param recursive: Если True, ищет файлы рекурсивно во всех подкаталогах.
    :param list_files: Если True, выводит список найденных файлов.
    :return: Количество найденных файлов.
    """
    directory = Path(directory)
    if not directory.is_dir():
        print(f"Ошибка: '{directory}' не является директорией или не существует.")
        return 0

    # Форматируем расширения для поиска
    extensions = [ext if ext.startswith('.') else f'.{ext}' for ext in extensions]

    if recursive:
        files = [f for f in directory.rglob('*') if f.suffix.lower() in extensions]
    else:
        files = [f for f in directory.glob('*') if f.suffix.lower() in extensions]

    count = len(files)

    if list_files:
        print(f"Найдено {count} файлов с расширениями {', '.join(extensions)}:")
        for file in files:
            print(file)

    return count

def main():
    parser = argparse.ArgumentParser(
        description="Подсчёт количества файлов с заданным расширением в директории."
    )
    parser.add_argument(
        '--dir', '-d',
        type=str,
        required=True,
        help='Путь к директории для проверки.'
    )
    parser.add_argument(
        '--ext', '-e',
        type=str,
        nargs='+',
        required=True,
        help='Расширения файлов для поиска (без точки). Пример: png jpg jpeg'
    )
    parser.add_argument(
        '--recursive', '-r',
        action='store_true',
        help='Включить рекурсивный поиск во всех подкаталогах.'
    )
    parser.add_argument(
        '--list', '-l',
        action='store_true',
        help='Вывести список найденных файлов.'
    )

    args = parser.parse_args()

    count = count_files(
        directory=args.dir,
        extensions=args.ext,
        recursive=args.recursive,
        list_files=args.list
    )

    print(f"\nИтого найдено {count} файлов(а) с расширениями {', '.join(['.' + ext for ext in args.ext])} в директории '{args.dir}'.")

if __name__ == "__main__":
    main()
