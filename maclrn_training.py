"""
Unified CAMPUS pipeline
- Downloads images from Supabase occupancy_logs
- Auto labels with YOLOv8 (people only)
- Splits dataset by images, ensuring train and test are from the same room
- Prepares YOLO folder layout
- Trains YOLOv8 on the auto-labeled dataset
- Evaluates on the 20 percent test set
- Produces detection-level metrics, image-level confusion matrix, and CSV reports

Usage
1) Put this file on your Jetson or Colab environment
2) Install dependencies: pip install supabase ultralytics opencv-python-headless pandas scikit-learn tqdm matplotlib
3) Run: python campus_pipeline.py

Notes
- Training uses device automatically detected, set FORCE_DEVICE environment variable to override
- Outputs are saved under ./outputs, including metrics.json, confusion_matrix.png, and a per-image CSV
"""

import os
import json
import time
import shutil
import requests
from tqdm import tqdm
import numpy as np
import pandas as pd
import cv2
import torch
from datetime import datetime
from sklearn.model_selection import train_test_split
from ultralytics import YOLO
from supabase import create_client
import matplotlib.pyplot as plt

# ---------------- CONFIG ----------------
SUPABASE_URL = "https://zpezidrqlotoyequnywe.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6InpwZXppZHJxbG90b3llcXVueXdlIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NjQyMTE5MzQsImV4cCI6MjA3OTc4NzkzNH0.mWBIhSNpmSaoPN1rytp9JlXdcv_kG9i6aNqBwxGo4q0"
BUCKET_NAME = "room_images"
LOGS_TABLE = "occupancy_logs"
ROOM_NAME = "GK407"  # keep for naming uploaded images, not strictly needed for download

# Where data and outputs are stored
BASE_DIR = "dataset"
IMG_ALL_DIR = os.path.join(BASE_DIR, "images_all")
LBL_ALL_DIR = os.path.join(BASE_DIR, "labels_all")
YOLO_IMAGES_TRAIN = os.path.join(BASE_DIR, "images/train")
YOLO_IMAGES_VAL = os.path.join(BASE_DIR, "images/val")
YOLO_LABELS_TRAIN = os.path.join(BASE_DIR, "labels/train")
YOLO_LABELS_VAL = os.path.join(BASE_DIR, "labels/val")
OUTPUT_DIR = "outputs"

# Training hyperparameters
EPOCHS = 30
IMGSZ = 640
BATCH = 8
IOU_THRESH = 0.5  # IoU threshold for matching detections to ground truth
CONF_THRESH = 0.25  # detection confidence threshold for counting detections

# Device selection
FORCE_DEVICE = os.environ.get("FORCE_DEVICE", None)
if FORCE_DEVICE:
    DEVICE = FORCE_DEVICE
else:
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

print("Device:", DEVICE)

# Create output folders
for d in [BASE_DIR, IMG_ALL_DIR, LBL_ALL_DIR, YOLO_IMAGES_TRAIN, YOLO_IMAGES_VAL, YOLO_LABELS_TRAIN, YOLO_LABELS_VAL, OUTPUT_DIR]:
    os.makedirs(d, exist_ok=True)

# ---------------- UTILITIES ----------------

def download_images_from_supabase(supabase_client, logs_table=LOGS_TABLE, limit=None):
    print("Fetching rows from Supabase table", logs_table)
    resp = supabase_client.table(logs_table).select("*").execute()
    rows = resp.data
    if limit:
        rows = rows[:limit]
    df = pd.DataFrame(rows)
    if df.empty:
        print("No rows found in table, returning empty list")
        return []

    if "image_url" not in df.columns:
        raise ValueError("Column 'image_url' not found in occupancy_logs table")

    df = df[df["image_url"].notnull()]
    df = df[df["image_url"] != ""]
    df = df.reset_index(drop=True)

    valid_image_paths = []
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Downloading images"):
        url = row["image_url"]
        if not isinstance(url, str) or len(url) < 10:
            continue
        img_name = f"image_{idx}.jpg"
        img_path = os.path.join(IMG_ALL_DIR, img_name)
        try:
            r = requests.get(url, timeout=15)
            if r.status_code != 200:
                continue
            with open(img_path, "wb") as f:
                f.write(r.content)
            img = cv2.imread(img_path)
            if img is None:
                os.remove(img_path)
                continue
            valid_image_paths.append(img_path)
        except Exception as e:
            print("Download error", e)

    print("Downloaded images:", len(valid_image_paths))
    return valid_image_paths


def auto_label_images(image_paths, model, labels_dir=LBL_ALL_DIR):
    """Auto label images using a YOLO model, writes YOLO format .txt labels in labels_dir"""
    os.makedirs(labels_dir, exist_ok=True)
    labeled_count = 0
    for img_path in tqdm(image_paths, desc="Auto labeling"):
        img = cv2.imread(img_path)
        if img is None:
            continue
        results = model.predict(img, classes=[0], verbose=False)  # person class only
        label_lines = []
        if len(results) > 0:
            r = results[0]
            if r.boxes is not None and getattr(r.boxes, "xywhn", None) is not None:
                for box in r.boxes.xywhn:
                    x, y, w, h = box[:4].tolist()
                    label_lines.append(f"0 {x} {y} {w} {h}")
        if len(label_lines) > 0:
            label_path = os.path.join(labels_dir, os.path.basename(img_path).replace('.jpg', '.txt'))
            with open(label_path, 'w') as f:
                f.write("\n".join(label_lines))
            labeled_count += 1
    print("Auto-labeled images:", labeled_count)
    return labeled_count


def collect_paired_images_and_labels(images_dir=IMG_ALL_DIR, labels_dir=LBL_ALL_DIR):
    images = sorted([os.path.join(images_dir, f) for f in os.listdir(images_dir) if f.lower().endswith('.jpg')])
    paired_images = []
    paired_labels = []
    for img in images:
        label = os.path.join(labels_dir, os.path.basename(img).replace('.jpg', '.txt'))
        if os.path.exists(label):
            paired_images.append(img)
            paired_labels.append(label)
    print("Paired images+labels:", len(paired_images))
    return paired_images, paired_labels


def prepare_yolo_folders(train_imgs, val_imgs, labels_dir=LBL_ALL_DIR):
    # Clean previous
    for d in [YOLO_IMAGES_TRAIN, YOLO_IMAGES_VAL, YOLO_LABELS_TRAIN, YOLO_LABELS_VAL]:
        os.makedirs(d, exist_ok=True)

    def copy_pairs(img_list, dst_img_dir, dst_lbl_dir):
        for img_path in img_list:
            img_name = os.path.basename(img_path)
            lbl_path = os.path.join(labels_dir, img_name.replace('.jpg', '.txt'))
            if os.path.exists(lbl_path):
                shutil.copy2(img_path, os.path.join(dst_img_dir, img_name))
                shutil.copy2(lbl_path, os.path.join(dst_lbl_dir, os.path.basename(lbl_path)))

    copy_pairs(train_imgs, YOLO_IMAGES_TRAIN, YOLO_LABELS_TRAIN)
    copy_pairs(val_imgs, YOLO_IMAGES_VAL, YOLO_LABELS_VAL)

    print('YOLO folders prepared:')
    print(' -', YOLO_IMAGES_TRAIN, len(os.listdir(YOLO_IMAGES_TRAIN)))
    print(' -', YOLO_IMAGES_VAL, len(os.listdir(YOLO_IMAGES_VAL)))

    # Write data.yaml
    data_yaml = f"""
train: {YOLO_IMAGES_TRAIN}
val: {YOLO_IMAGES_VAL}

nc: 1
names: ['person']
"""
    with open('data.yaml', 'w') as f:
        f.write(data_yaml)
    print('Wrote data.yaml')


def train_yolo(data_yaml='data.yaml', epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, pretrained='yolov8n.pt'):
    print('Starting training, this may take a while')
    model = YOLO(pretrained)
    model.train(
        data=data_yaml,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=DEVICE
    )
    # After training the best model is usually saved under runs/train/exp or similar
    print('Training finished')
    return model

# ---------------- EVALUATION ----------------

def xywhn_to_xyxy(xywhn, img_w, img_h):
    # input is [x_center_norm, y_center_norm, w_norm, h_norm]
    x, y, w, h = xywhn
    xc = x * img_w
    yc = y * img_h
    bw = w * img_w
    bh = h * img_h
    x1 = xc - bw / 2.0
    y1 = yc - bh / 2.0
    x2 = xc + bw / 2.0
    y2 = yc + bh / 2.0
    return [x1, y1, x2, y2]


def iou(boxA, boxB):
    # boxes in [x1,y1,x2,y2]
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, boxA[2] - boxA[0]) * max(0, boxA[3] - boxA[1])
    boxBArea = max(0, boxB[2] - boxB[0]) * max(0, boxB[3] - boxB[1])
    unionArea = boxAArea + boxBArea - interArea
    if unionArea == 0:
        return 0.0
    return interArea / unionArea


def read_yolo_labels(label_path, img_w, img_h):
    boxes = []
    with open(label_path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls = int(parts[0])
            xywhn = list(map(float, parts[1:5]))
            xyxy = xywhn_to_xyxy(xywhn, img_w, img_h)
            boxes.append({'cls': cls, 'bbox': xyxy})
    return boxes


def evaluate_on_test(model_path, test_img_dir, test_lbl_dir, iou_thresh=IOU_THRESH, conf_thresh=CONF_THRESH):
    print('Loading model for evaluation:', model_path)
    model = YOLO(model_path)
    model.to(DEVICE)

    image_files = sorted([f for f in os.listdir(test_img_dir) if f.lower().endswith('.jpg')])

    # Detection-level counters
    total_TP = 0
    total_FP = 0
    total_FN = 0

    # Image-level confusion matrix counters
    img_TP = 0
    img_FP = 0
    img_FN = 0
    img_TN = 0

    per_image_records = []

    for img_name in tqdm(image_files, desc='Evaluating test images'):
        img_path = os.path.join(test_img_dir, img_name)
        lbl_path = os.path.join(test_lbl_dir, img_name.replace('.jpg', '.txt'))

        img = cv2.imread(img_path)
        if img is None:
            continue
        h, w = img.shape[:2]

        gt_boxes = []
        if os.path.exists(lbl_path):
            gt_boxes = read_yolo_labels(lbl_path, w, h)

        results = model.predict(img, imgsz=IMGSZ, verbose=False)[0]

        preds = []
        if getattr(results, 'boxes', None) is not None:
            for b in results.boxes:
                conf = float(b.conf) if getattr(b, 'conf', None) is not None else float(b[4])
                clsid = int(b.cls) if getattr(b, 'cls', None) is not None else int(b[5])
                xyxy = b.xyxy.tolist() if hasattr(b, 'xyxy') else b[:4].tolist()
                if conf >= conf_thresh and clsid == 0:
                    preds.append({'cls': clsid, 'bbox': xyxy, 'conf': conf})

        # Detection-level matching
        matched_gt = set()
        matched_pred = set()

        for pi, p in enumerate(preds):
            best_iou = 0.0
            best_gi = -1
            for gi, g in enumerate(gt_boxes):
                if gi in matched_gt:
                    continue
                current_iou = iou(p['bbox'], g['bbox'])
                if current_iou > best_iou:
                    best_iou = current_iou
                    best_gi = gi
            if best_iou >= iou_thresh:
                total_TP += 1
                matched_gt.add(best_gi)
                matched_pred.add(pi)
            else:
                total_FP += 1

        # Any unmatched gt are false negatives
        total_FN += (len(gt_boxes) - len(matched_gt))

        # Image-level presence confusion
        gt_has = 1 if len(gt_boxes) > 0 else 0
        pred_has = 1 if len(preds) > 0 else 0
        if gt_has == 1 and pred_has == 1:
            img_TP += 1
        elif gt_has == 0 and pred_has == 1:
            img_FP += 1
        elif gt_has == 1 and pred_has == 0:
            img_FN += 1
        else:
            img_TN += 1

        per_image_records.append({
            'image': img_name,
            'gt_count': len(gt_boxes),
            'pred_count': len(preds),
            'matched_gt': len(matched_gt),
            'unmatched_gt': len(gt_boxes) - len(matched_gt),
            'fp': len(preds) - len(matched_gt)
        })

    # Compute detection metrics
    precision = total_TP / (total_TP + total_FP) if (total_TP + total_FP) > 0 else 0.0
    recall = total_TP / (total_TP + total_FN) if (total_TP + total_FN) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Image-level confusion matrix and derived metrics
    cm = np.array([[img_TN, img_FP], [img_FN, img_TP]])
    img_accuracy = (img_TP + img_TN) / cm.sum() if cm.sum() > 0 else 0.0
    img_precision = img_TP / (img_TP + img_FP) if (img_TP + img_FP) > 0 else 0.0
    img_recall = img_TP / (img_TP + img_FN) if (img_TP + img_FN) > 0 else 0.0
    img_f1 = 2 * img_precision * img_recall / (img_precision + img_recall) if (img_precision + img_recall) > 0 else 0.0

    metrics = {
        'detection': {
            'TP': int(total_TP),
            'FP': int(total_FP),
            'FN': int(total_FN),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        },
        'image_level': {
            'TP': int(img_TP),
            'FP': int(img_FP),
            'FN': int(img_FN),
            'TN': int(img_TN),
            'accuracy': float(img_accuracy),
            'precision': float(img_precision),
            'recall': float(img_recall),
            'f1': float(img_f1)
        }
    }

    # Save outputs
    with open(os.path.join(OUTPUT_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    pd.DataFrame(per_image_records).to_csv(os.path.join(OUTPUT_DIR, 'per_image_results.csv'), index=False)

    # Plot image-level confusion matrix
    plt.figure(figsize=(5, 4))
    plt.imshow(cm, interpolation='nearest')
    plt.title('Image-level confusion matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ['NoPerson', 'Person'])
    plt.yticks(tick_marks, ['NoPerson', 'Person'])

    thresh = cm.max() / 2. if cm.max() > 0 else 0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'), horizontalalignment='center', verticalalignment='center')

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, 'confusion_matrix.png'))
    plt.close()

    print('Evaluation complete, metrics saved to', OUTPUT_DIR)
    return metrics

# ---------------- MAIN PIPELINE ----------------

def main():
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)

    # 1) Download images
    images = download_images_from_supabase(supabase)
    if len(images) == 0:
        print('No images to process, exiting')
        return

    # 2) Auto-label using yolov8n
    labeller = YOLO('yolov8n.pt')
    auto_label_images(images, labeller)

    # 3) Collect paired images and labels
    paired_images, paired_labels = collect_paired_images_and_labels()
    if len(paired_images) == 0:
        print('No paired images and labels found, exiting')
        return

    # 4) Split 80/20 ensuring same-room images are used, because all images come from the same supabase room table
    train_imgs, test_imgs, train_lbls, test_lbls = train_test_split(
        paired_images, paired_labels, test_size=0.2, random_state=42
    )

    # 5) Prepare YOLO folders
    prepare_yolo_folders(train_imgs, test_imgs)

    # 6) Train YOLO on prepared dataset
    trained_model = train_yolo(data_yaml='data.yaml', epochs=EPOCHS, imgsz=IMGSZ, batch=BATCH, pretrained='yolov8n.pt')

    # Try to locate the best saved weights, otherwise use the trained model object
    # ultralytics saves weights under runs/train/exp or runs/train/expN. We will try to find the latest
    runs_dir = 'runs/train'
    best_weights = None
    if os.path.exists(runs_dir):
        # find most recent exp folder
        exps = sorted([os.path.join(runs_dir, d) for d in os.listdir(runs_dir)], key=os.path.getmtime, reverse=True)
        if exps:
            latest = exps[0]
            # common weight names
            candidates = ['best.pt', 'weights/best.pt', 'weights/last.pt', 'last.pt']
            for c in candidates:
                path = os.path.join(latest, c)
                if os.path.exists(path):
                    best_weights = path
                    break
    if best_weights is None:
        # fallback to using ultralytics model object file if available
        # the YOLO object can be passed directly to evaluation functions, but our evaluate function expects a path
        print('No saved weights found, attempting to use the model object directly for evaluation')
        # ultralytics allows model.export or model.save, but for simplicity we will save last.pt
        save_path = os.path.join(OUTPUT_DIR, 'trained_last.pt')
        try:
            trained_model.export(format='pt')
            # exported file name depends on ultralytics behavior, fallback to the save path
            best_weights = save_path if os.path.exists(save_path) else None
        except Exception:
            best_weights = None

    # If we still have no weight path, try default 'yolov8n.pt' as safety
    if best_weights is None:
        print('Could not find trained best weights, using the pretrained yolov8n.pt for evaluation')
        best_weights = 'yolov8n.pt'

    # 7) Evaluate on test set
    metrics = evaluate_on_test(best_weights, YOLO_IMAGES_VAL, YOLO_LABELS_VAL)

    print('\nSummary metrics:')
    print(json.dumps(metrics, indent=2))

if __name__ == '__main__':
    main()
