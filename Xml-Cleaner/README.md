---
title: Intelligent XML Cleaner
emoji: 🌳
colorFrom: blue
colorTo: green
sdk: gradio
sdk_version: 6.1.0
app_file: app.py
pinned: false
---
# Intelligent XML Cleaner & Visualizer

This tool helps Android developers and QA engineers clean stale accessibility node information from UI XML dumps.

## Features

* **Active-Based Sibling Pruning:** Intelligently removes XML nodes that are not visible on the screen based on OCR analysis or manual text input.
* **Flexible Text Input:** Optionally provide visible text manually, or use OCR for automatic extraction.
* **Dual OCR Strategy:** Choose between **EasyOCR** (Deep Learning based, high accuracy) or **Tesseract** (Fast, standard) as fallback when manual text is not provided.
* **Comprehensive Visualization:**
  * **Tree View:** See the hierarchical structure of your XML before and after cleaning.
  * **Screen View:** Visual confirmation of bounding boxes overlaid on the original screenshot.

## How to use

1. Upload the Screenshot of the app state.
2. Upload the corresponding XML dump (from `uiautomator`).
3. **(Optional)** Enter visible text from the screenshot manually (one per line or comma-separated). If left empty, OCR will be used automatically.
4. Select your preferred OCR engine (only used if visible text is not provided).
5. Click **Process**.
6. View the comparisons in the tabs and download the cleaned XML.

## Technical Details

This application uses a sophisticated pipeline:

1. **Text Extraction:** Uses provided visible text (if available) or extracts visible text from the image using OCR.
2. **LCA Calculation:** Finds the Lowest Common Ancestor of all active elements.
3. **Pruning:** Traverses upward from the Active LCA and prunes siblings that contain no visible text.

