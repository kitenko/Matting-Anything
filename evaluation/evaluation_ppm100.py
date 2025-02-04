import os
import sys
import cv2
import numpy as np
sys.path.insert(0, './utils')
from   evaluate import compute_sad_loss, compute_mse_loss, compute_mad_loss
import argparse
from tqdm import tqdm

# Список поддерживаемых форматов изображений
VALID_IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif']

def is_image(file_name):
    """Проверка, является ли файл изображением на основе расширения."""
    return any(file_name.lower().endswith(ext) for ext in VALID_IMAGE_EXTENSIONS)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred-dir', type=str, default='path/to/outputs/ppm100', help="pred alpha dir")
    parser.add_argument('--label-dir', type=str, default='path/to/PPM-100/matte', help="GT alpha dir")
    parser.add_argument('--detailmap-dir', type=str, default='path/to/PPM-100/matte', help="trimap dir")

    args = parser.parse_args()

    mse_loss = []
    sad_loss = []
    mad_loss = []
    ### loss_unknown only consider the unknown regions, i.e. trimap==128, as trimap-based methods do
    #mse_loss_unknown = []
    #sad_loss_unknown = []
    
    for img in tqdm(os.listdir(args.label_dir)):
        if not is_image(img):
            continue
        print(img)
        #pred = cv2.imread(os.path.join(args.pred_dir, img.replace('.png', '.jpg')), 0).astype(np.float32)
        pred = cv2.imread(os.path.join(args.pred_dir, img.replace('.jpg', '.png')), 0).astype(np.float32)
        label = cv2.imread(os.path.join(args.label_dir, img), 0).astype(np.float32)
        detailmap = cv2.imread(os.path.join(args.detailmap_dir, img), 0).astype(np.float32)

        #detailmap[detailmap > 0] = 128

        #mse_loss_unknown_ = compute_mse_loss(pred, label, detailmap)
        #sad_loss_unknown_ = compute_sad_loss(pred, label, detailmap)[0]

        detailmap[...] = 128

        mse_loss_ = compute_mse_loss(pred, label, detailmap)
        sad_loss_ = compute_sad_loss(pred, label, detailmap)[0]
        mad_loss_ = compute_mad_loss(pred, label, detailmap)
        
        print('Whole Image: MSE:', mse_loss_, ' SAD:', sad_loss_, ' MAD:', mad_loss_)
        #print('Detail Region: MSE:', mse_loss_unknown_, ' SAD:', sad_loss_unknown_)

        #mse_loss_unknown.append(mse_loss_unknown_)
        #sad_loss_unknown.append(sad_loss_unknown_)

        mse_loss.append(mse_loss_)
        sad_loss.append(sad_loss_)
        mad_loss.append(mad_loss_)

    print('Average:')
    print('Whole Image: MSE:', np.array(mse_loss).mean(), ' SAD:', np.array(sad_loss).mean(), ' MAD:', np.array(mad_loss).mean())
    #print('Detail Region: MSE:', np.array(mse_loss_unknown).mean(), ' SAD:', np.array(sad_loss_unknown).mean())
