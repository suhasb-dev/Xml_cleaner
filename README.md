# Clean XML - Stale Element Removal Tool

A Python tool that removes stale (invisible) screen elements from Android XML dumps by comparing them with OCR-detected visible text from screenshots.

## Overview

This tool uses an **Active-Based Sibling Pruning** approach to clean XML files:
1. Extracts visible text from screenshots using OCR
2. Identifies active elements (matching OCR text) and stale elements (not visible)
3. Finds the Lowest Common Ancestor (LCA) of all active elements
4. Traverses upward from the active LCA, pruning sibling subtrees containing stale elements
5. Outputs a cleaned XML file with only visible elements

## Features

- ✅ Automatic detection of visible vs. invisible elements
- ✅ Semantic text matching (handles OCR variations)
- ✅ Efficient subtree pruning algorithm
- ✅ Latency measurement for performance monitoring
- ✅ Preserves all active elements and their hierarchy

## Installation

### Requirements

- Python 3.7+
- EasyOCR
- OpenCV (cv2)
- Standard library: `pathlib`, `xml.etree.ElementTree`, `re`, `time`

### Install Dependencies

```bash
pip install easyocr opencv-python
```

## Usage

### Quick Start

```bash
python clean_xml_active_lca.py
```

### Programmatic Usage

```python
from clean_xml_active_lca import XMLCleaner

cleaner = XMLCleaner()
result = cleaner.clean(
    image_path="path/to/screenshot.png",
    xml_path="path/to/dump.xml",
    output_path="path/to/cleaned.xml"
)

print(f"Total Latency: {result['total_latency']:.4f} seconds")
print(f"Elements removed: {result['removed']}")
```

## Input/Output

### Input
- **Image**: Screenshot of the Android screen (PNG, JPG, etc.)
- **XML**: Android UI hierarchy dump (XML format)

### Output
- **Cleaned XML**: XML file with stale elements removed
- **Location**: Saved to `/active_output/` directory by default

## Latency Metrics

The tool provides detailed latency breakdown:

- **OCR Latency**: Time for text extraction from image (usually the bottleneck)
- **XML Cleaning Latency**: Time for XML processing only (~0.004 seconds)
- **TOTAL Latency**: Complete end-to-end time (~20 seconds typical)

**For production reporting, use TOTAL Latency** as it represents the full user experience.

## Algorithm

### Active-Based Sibling Pruning

1. **OCR Extraction**: Extract visible text from screenshot
2. **Active Detection**: Find XML elements whose text matches OCR results (semantic matching with 50% token overlap)
3. **LCA Finding**: Find the Lowest Common Ancestor of all active elements
4. **Upward Traversal**: Starting from active LCA, traverse upward to root
5. **Sibling Pruning**: At each level, check siblings and remove subtrees containing stale elements
6. **Output**: Save cleaned XML preserving active element hierarchy

### Why This Approach?

- **Deterministic**: No uncertainty - only prunes siblings that definitely contain stale elements
- **Safe**: Preserves all active elements and their relationships
- **Efficient**: O(n) complexity for tree traversal
- **Optimal**: Removes maximum stale content while maintaining structure

## File Structure

```
Clean_Xml/
├── clean_xml_active_lca.py    # Main cleaning script (simplified, for latency measurement)
├── clean_xml.py                # Full version with visualizations
├── xml_tree_visualizer.py      # Tree visualization tool
├── data/                       # Input files (images, XML)
├── active_output/              # Output directory for cleaned XML
└── README.md                   # This file
```

## Example Output

```
============================================================
✅ RESULTS
============================================================
Status:           cleaned
Elements removed: 231
Active elements:  45
Stale elements:   186

------------------------------------------------------------
⏱️  LATENCY BREAKDOWN
------------------------------------------------------------
OCR Latency:           19.5234 seconds
XML Cleaning Latency:  0.0041 seconds
TOTAL Latency:         19.5275 seconds

💡 For production reporting, use TOTAL Latency

💾 Cleaned XML saved to: /active_output/Giczg_cleaned.xml
============================================================
```

## Performance

- **XML Cleaning**: ~0.004 seconds (very fast)
- **OCR Processing**: ~15-20 seconds (depends on image complexity)
- **Total End-to-End**: ~20 seconds typical

## Notes

- OCR is the main bottleneck - consider GPU acceleration for faster processing
- Semantic matching handles OCR variations (e.g., "Hello" matches "hello", "Hello World")
- Empty text nodes are preserved in the tree structure
- All active elements and their parent-child relationships are maintained

## License

[Add your license here]

## Author

[Add your name/contact here]

