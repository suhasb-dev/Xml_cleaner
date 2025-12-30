from abc import ABC, abstractmethod
from typing import Set

# Lazy imports to handle missing dependencies gracefully
try:
    import easyocr
    EASYOCR_AVAILABLE = True
except ImportError:
    EASYOCR_AVAILABLE = False

try:
    import pytesseract
    PYTESSERACT_AVAILABLE = True
except ImportError:
    PYTESSERACT_AVAILABLE = False

try:
    import cv2
    import numpy as np
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False

class OCRStrategy(ABC):
    """Abstract base class for OCR strategies."""
    @abstractmethod
    def extract_text(self, image_path: str) -> Set[str]:
        pass

class EasyOCRStrategy(OCRStrategy):
    """Concrete strategy for EasyOCR."""
    def __init__(self):
        if not EASYOCR_AVAILABLE:
            raise ImportError(
                "EasyOCR is not installed. Please install it with: pip install easyocr"
            )
        # Initialize once to save memory/time
        print("Loading EasyOCR Model...")
        self.reader = easyocr.Reader(['en'], gpu=False)

    def extract_text(self, image_path: str) -> Set[str]:
        results = self.reader.readtext(image_path, detail=0)
        return {text.lower().strip() for text in results if text.strip()}

class TesseractOCRStrategy(OCRStrategy):
    """Concrete strategy for Tesseract OCR (Free & Fast)."""
    def __init__(self):
        if not PYTESSERACT_AVAILABLE:
            raise ImportError(
                "pytesseract is not installed. Please install it with: pip install pytesseract"
            )
        if not CV2_AVAILABLE:
            raise ImportError(
                "opencv-python is not installed. Please install it with: pip install opencv-python-headless"
            )
    
    def extract_text(self, image_path: str) -> Set[str]:
        # Preprocessing for better Tesseract accuracy
        img = cv2.imread(image_path)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Apply thresholding
        _, thresh = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)
        
        # Tesseract execution
        text = pytesseract.image_to_string(thresh)
        
        # Process results
        lines = text.split('\n')
        return {line.lower().strip() for line in lines if line.strip()}

class OCRFactory:
    """Factory to get the correct OCR strategy."""
    _instances = {}

    @staticmethod
    def get_strategy(strategy_name: str) -> OCRStrategy:
        if strategy_name not in OCRFactory._instances:
            if strategy_name == "EasyOCR (Best Accuracy)":
                OCRFactory._instances[strategy_name] = EasyOCRStrategy()
            elif strategy_name == "Tesseract (Fast & Free)":
                OCRFactory._instances[strategy_name] = TesseractOCRStrategy()
            else:
                raise ValueError(f"Unknown strategy: {strategy_name}")
        return OCRFactory._instances[strategy_name]