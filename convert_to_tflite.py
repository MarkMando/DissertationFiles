"""
=============================================================================
External Image Module — ONNX → TFLite Conversion
Model   : ext_mobilenetv3.onnx
Output  : Float32 TFLite + INT8 quantised TFLite
Author  : Mark Sthembiso Mando | Mulungushi University MSc Data Science
=============================================================================

ENVIRONMENT — run in tflite_convert conda env (NOT breastcancer):
    conda create -n tflite_convert python=3.10 -y
    conda activate tflite_convert
    pip install tensorflow tf_keras onnx onnx2tf onnx-graphsurgeon sng4onnx psutil pillow numpy

HOW TO RUN:
    python convert_to_tflite.py

NOTE: This script uses PIL only — no PyTorch or torchvision required.
      Keeps the breastcancer training environment untouched.
=============================================================================
"""

import os
import glob
import time
import warnings
import numpy as np
warnings.filterwarnings("ignore")
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"   # suppress TF C++ info logs

from PIL import Image
import tensorflow as tf
import onnx2tf

# =============================================================================
# CONFIGURATION
# =============================================================================

DATASET_DIR  = r'C:\Users\mmando_Adm\Documents\final_synthetic_dataset'
OUTPUT_DIR   = r'C:\Users\mmando_Adm\Documents\external_image_output'
ONNX_PATH    = os.path.join(OUTPUT_DIR, "ext_mobilenetv3.onnx")
TFLITE_DIR   = os.path.join(OUTPUT_DIR, "tflite")
F32_DIR      = os.path.join(TFLITE_DIR, "float32")
INT8_DIR     = os.path.join(TFLITE_DIR, "int8")
F32_PATH     = os.path.join(F32_DIR,  "ext_mobilenetv3_f32.tflite")
INT8_PATH    = os.path.join(INT8_DIR, "ext_mobilenetv3_int8.tflite")

IMG_SIZE     = 224

# ImageNet normalisation constants
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD  = np.array([0.229, 0.224, 0.225], dtype=np.float32)

os.makedirs(F32_DIR,  exist_ok=True)
os.makedirs(INT8_DIR, exist_ok=True)

print(f"\n{'='*62}")
print("  ONNX → TFLite  |  External Image Module")
print(f"{'='*62}\n")
print(f"  Source  : {ONNX_PATH}")
print(f"  Output  : {TFLITE_DIR}\n")

if not os.path.exists(ONNX_PATH):
    raise FileNotFoundError(
        f"ONNX file not found: {ONNX_PATH}\n"
        "Run train_synthetic_external.py first to generate it.")


# =============================================================================
# IMAGE PREPROCESSING (PIL — no torchvision)
# =============================================================================

def preprocess_image(img_path: str) -> np.ndarray:
    """
    Loads and preprocesses a single image to match training transforms:
      Resize → (224, 224) | ToTensor → [0,1] | Normalise (ImageNet)
    Returns numpy array shape (1, 224, 224, 3) channels-last (NHWC).
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)
    arr = np.array(img, dtype=np.float32) / 255.0   # [0, 1]
    arr = (arr - MEAN) / STD                         # ImageNet normalise
    return arr[np.newaxis, ...]                      # (1, H, W, 3)


def get_val_images(max_images: int = 150) -> list:
    """Returns list of image paths from the val split."""
    pattern = os.path.join(DATASET_DIR, "val", "**", "*.png")
    paths   = glob.glob(pattern, recursive=True)
    if not paths:   # try jpg if no png
        pattern = os.path.join(DATASET_DIR, "val", "**", "*.jpg")
        paths   = glob.glob(pattern, recursive=True)
    return paths[:max_images]


# =============================================================================
# STEP 1 — ONNX → TFLite Float32
# =============================================================================

print("─── Step 1 / 3 : Float32 conversion ───")

onnx2tf.convert(
    input_onnx_file_path = ONNX_PATH,
    output_folder_path   = F32_DIR,
    non_verbose          = True,
    verbosity            = "error",
)

# onnx2tf generates the file with a derived name — find and rename
tflite_files = [f for f in os.listdir(F32_DIR) if f.endswith(".tflite")]
if not tflite_files:
    raise FileNotFoundError(
        "onnx2tf did not produce a .tflite file. "
        "Check onnx2tf output above for errors.")

generated = os.path.join(F32_DIR, tflite_files[0])
if generated != F32_PATH:
    os.rename(generated, F32_PATH)

f32_mb = os.path.getsize(F32_PATH) / (1024 ** 2)
print(f"  ✓ Saved → {F32_PATH}  ({f32_mb:.2f} MB)\n")


# =============================================================================
# STEP 2 — INT8 Post-Training Quantisation
#    Representative dataset: val split preprocessed with PIL.
#    I/O kept as float32 for Flutter/Android compatibility.
# =============================================================================

print("─── Step 2 / 3 : INT8 quantised TFLite ───")

val_paths = get_val_images()
if not val_paths:
    raise FileNotFoundError(
        f"No images found in {DATASET_DIR}/val/. "
        "Check DATASET_DIR in configuration.")

print(f"  Calibrating on {len(val_paths)} val images ...")

def representative_dataset():
    for path in val_paths:
        try:
            yield [preprocess_image(path)]
        except Exception:
            continue   # skip corrupt images silently

# Locate SavedModel — check F32_DIR itself first, then subdirectories.
# Skips 'assets'/'variables' folders that don't contain saved_model.pb.
def find_saved_model(base_dir: str) -> str:
    if os.path.exists(os.path.join(base_dir, "saved_model.pb")):
        return base_dir
    for name in os.listdir(base_dir):
        candidate = os.path.join(base_dir, name)
        if os.path.isdir(candidate):
            if os.path.exists(os.path.join(candidate, "saved_model.pb")):
                return candidate
    raise RuntimeError(
        f"No SavedModel (saved_model.pb) found in {base_dir}. "
        "Step 1 may not have completed successfully.")

saved_model_path = find_saved_model(F32_DIR)
print(f"  SavedModel : {saved_model_path}")

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
converter.optimizations             = [tf.lite.Optimize.DEFAULT]
converter.representative_dataset    = representative_dataset
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type      = tf.float32   # keep float32 I/O
converter.inference_output_type     = tf.float32

tflite_int8 = converter.convert()

with open(INT8_PATH, 'wb') as f:
    f.write(tflite_int8)

int8_mb = os.path.getsize(INT8_PATH) / (1024 ** 2)
print(f"  ✓ Saved → {INT8_PATH}  ({int8_mb:.2f} MB)\n")


# =============================================================================
# STEP 3 — Latency benchmark (CPU, 50 runs each)
# =============================================================================

print("─── Step 3 / 3 : Latency benchmark ───")

def benchmark(tflite_path: str, label: str) -> float:
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp_det = interp.get_input_details()[0]
    dummy   = np.random.rand(1, IMG_SIZE, IMG_SIZE, 3).astype(np.float32)

    for _ in range(5):   # warmup
        interp.set_tensor(inp_det['index'], dummy)
        interp.invoke()

    times = []
    for _ in range(50):
        interp.set_tensor(inp_det['index'], dummy)
        t0 = time.perf_counter()
        interp.invoke()
        times.append((time.perf_counter() - t0) * 1000)

    mean_ms = float(np.mean(times))
    p95_ms  = float(np.percentile(times, 95))
    print(f"  {label:<22}  Mean: {mean_ms:5.1f} ms  P95: {p95_ms:5.1f} ms")
    return mean_ms

f32_ms  = benchmark(F32_PATH,  "Float32 TFLite")
int8_ms = benchmark(INT8_PATH, "INT8 TFLite")


# =============================================================================
# SUMMARY
# =============================================================================

print(f"\n{'='*62}")
print("  CONVERSION COMPLETE")
print(f"{'='*62}")
print(f"\n  {'Model':<22} {'Size':>8}  {'Latency':>10}  {'≤15MB':>6}  {'≤100ms':>7}")
print(f"  {'-'*58}")
print(f"  {'Float32 TFLite':<22} {f32_mb:>6.2f} MB  {f32_ms:>7.1f} ms"
      f"  {'✓' if f32_mb  <= 15 else '✗':>7}   {'✓' if f32_ms  <= 100 else '✗':>7}")
print(f"  {'INT8 TFLite':<22} {int8_mb:>6.2f} MB  {int8_ms:>7.1f} ms"
      f"  {'✓' if int8_mb <= 15 else '✗':>7}   {'✓' if int8_ms <= 100 else '✗':>7}")
print(f"\n  Recommended for deployment : INT8 TFLite")
print(f"\n  Android / Flutter integration:")
print(f"    1. Copy ext_mobilenetv3_int8.tflite → app/src/main/assets/")
print(f"    2. Input  : float32 [1, {IMG_SIZE}, {IMG_SIZE}, 3]"
      f"  (NHWC, ImageNet normalised)")
print(f"    3. Output : float32 [1, 2]"
      f"  → argmax → 0=abnormal / 1=normal")
print(f"{'='*62}\n")
