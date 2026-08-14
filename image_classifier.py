import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.applications.mobilenet_v2 import (
    MobileNetV2,
    preprocess_input,
    decode_predictions
)

from tensorflow.keras.preprocessing import image

import tkinter as tk
from tkinter import filedialog


# ==========================================
# 1. LOAD THE AI MODEL
# ==========================================

print("========================================")
print("       AI IMAGE CLASSIFICATION")
print("========================================")

print("\nLoading MobileNetV2 Model...")

# MobileNetV2 is a pre-trained image classification
# model trained on the ImageNet dataset.

model = MobileNetV2(weights="imagenet")

print("Model Loaded Successfully!")


# ==========================================
# 2. OPEN FILE SELECTION WINDOW
# ==========================================

print("\nPlease select an image...")

root = tk.Tk()
root.withdraw()

img_path = filedialog.askopenfilename(
    title="Select an Image",
    filetypes=[
        ("Image Files", "*.jpg *.jpeg *.png"),
        ("JPG Files", "*.jpg"),
        ("PNG Files", "*.png")
    ]
)


# ==========================================
# 3. CHECK IF IMAGE WAS SELECTED
# ==========================================

if not img_path:
    print("\nNo image selected.")
    print("Program ended.")
    exit()

print("\nSelected Image:")
print(img_path)


# ==========================================
# 4. LOAD IMAGE
# ==========================================

img = image.load_img(
    img_path,
    target_size=(224, 224)
)


# ==========================================
# 5. DISPLAY IMAGE
# ==========================================

plt.figure(figsize=(6, 6))
plt.imshow(img)
plt.title("Selected Image")
plt.axis("off")
plt.show()


# ==========================================
# 6. CONVERT IMAGE TO ARRAY
# ==========================================

img_array = image.img_to_array(img)


# ==========================================
# 7. ADD BATCH DIMENSION
# ==========================================

img_array = np.expand_dims(
    img_array,
    axis=0
)


# ==========================================
# 8. PREPROCESS IMAGE
# ==========================================

img_array = preprocess_input(img_array)


# ==========================================
# 9. MAKE AI PREDICTION
# ==========================================

print("\nPredicting...")

predictions = model.predict(img_array, verbose=0)


# ==========================================
# 10. GET TOP 5 PREDICTIONS
# ==========================================

results = decode_predictions(
    predictions,
    top=5
)[0]


# ==========================================
# 11. DISPLAY RESULTS
# ==========================================

print("\n========================================")
print("          TOP 5 PREDICTIONS")
print("========================================")

for i, (imagenet_id, label, probability) in enumerate(results):
    print(
        f"{i + 1}. {label} : {probability * 100:.2f}%"
    )


# ==========================================
# 12. BEST PREDICTION
# ==========================================

best_label = results[0][1]
best_probability = results[0][2] * 100

print("\n========================================")
print("          FINAL PREDICTION")
print("========================================")

print("Prediction :", best_label)
print(f"Confidence : {best_probability:.2f}%")

print("\n========================================")
print("     Prediction Completed Successfully!")
print("========================================")