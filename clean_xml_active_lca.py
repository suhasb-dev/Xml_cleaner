"""
Removes stale screen elements from XML using Active-Based Sibling Pruning approach:
1. Find active elements (visible in OCR) and stale elements (invisible)
2. Find LCA of all active elements
3. Traverse upward from active LCA to root
4. At each level, check siblings and prune subtrees containing stale elements

Simplified version: Image + XML → Cleaned XML (for latency measurement)
"""

# Standard library
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import xml.etree.ElementTree as ET
import time

# Third-party (install: pip install easyocr)
import easyocr
import re

# ============================================================================
# 1.Data Loading
# ============================================================================
class DataLoader:
    """Loads and validates image and XML inputs"""
    
    def load_inputs(self, image_path: str, xml_path: str) -> Tuple[Path, ET.ElementTree]:
        """
        Load image and XML files
        
        Args:
            image_path: Path to screenshot
            xml_path: Path to XML dump
            
        Returns:
            (image_path, xml_tree)
        """
        img_path = Path(image_path)
        xml_file = Path(xml_path)
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not xml_file.exists():
            raise FileNotFoundError(f"XML not found: {xml_path}")
        
        tree = ET.parse(xml_file)
        
        return img_path, tree

# ============================================================================
#2: OCR Extraction
# ============================================================================
class OCRExtractor:
    """Extracts visible text from screenshot using EasyOCR"""
    
    def __init__(self):
        self.reader = easyocr.Reader(['en'], gpu=False)  # Initialize EasyOCR reader with English, CPU mode
    
    def extract_visible_text(self, image_path: Path) -> Set[str]:
        """
        Extract all visible text from image using OCR
        
        Args:
            image_path: Path to screenshot
            
        Returns:
            Set of visible text strings (lowercased, stripped)
        """
        results = self.reader.readtext(str(image_path), detail=0)  # Read text from image, detail=0 returns only text
        
        visible_text = {text.lower().strip() for text in results if text.strip()}  # Lowercase and strip whitespace, filter empty strings
        
        return visible_text

# ============================================================================
#  3: XML Parsing
# ============================================================================
class XMLParser:
    """Parses XML and builds parent-child relationships"""
    
    def parse_xml(self, tree: ET.ElementTree) -> Tuple[ET.Element, Dict[ET.Element, ET.Element]]:
        """
        Parse XML tree and build parent map
        
        Args:
            tree: Parsed XML tree
            
        Returns:
            (root_element, parent_map)
        """
        root = tree.getroot()  # Get root element of XML tree
        
        parent_map = {child: parent for parent in root.iter() for child in parent}  # Build dict mapping each child to its parent
        
        return root, parent_map

# ============================================================================
#  4: Active & Stale Detection (UPDATED)
# ============================================================================
class StaleDetector:
    """Identifies active and stale elements by comparing XML with OCR results"""
    
    def find_active_and_stale(
        self, 
        root: ET.Element, 
        visible_text: Set[str]
    ) -> Tuple[List[ET.Element], List[ET.Element]]:
        """
        Find both active (visible) and stale (invisible) elements
        
        Args:
            root: XML root element
            visible_text: Set of visible text from OCR
            
        Returns:
            (active_elements, stale_elements)
        """
        active_elements = []
        stale_elements = []
        filtered_ocr = {t for t in visible_text if len(t) > 2}
        
        for elem in root.iter():
            text = elem.get('text', '').lower().strip()
            
            if text and len(text) > 2:
                if self._is_similar(text, filtered_ocr):
                    active_elements.append(elem)
                else:
                    stale_elements.append(elem)
        
        return active_elements, stale_elements
    
    def _is_similar(self, elem_text: str, visible_text: Set[str]) -> bool:
        """Check if text matches any visible text (semantic)"""
        elem_tokens = set(re.findall(r'\w+', elem_text))  # Tokenize
        
        for ocr_text in visible_text:
            ocr_tokens = set(re.findall(r'\w+', ocr_text))
            
            if elem_tokens & ocr_tokens:  # Any token overlap
                overlap_ratio = len(elem_tokens & ocr_tokens) / len(elem_tokens)
                if overlap_ratio >= 0.5:  # 50% match
                    return True
        
        return False

# ============================================================================
#  5: LCA (Lowest Common Ancestor) Finder
# ============================================================================

class LCAFinder:
    """Finds lowest common ancestor of elements"""
    
    def find_lca(self, elements: List[ET.Element], parent_map: Dict[ET.Element, ET.Element]) -> Optional[ET.Element]:
        """
        Find lowest common ancestor of given elements
        
        Args:
            elements: List of elements (can be active or stale)
            parent_map: Dict mapping child to parent
            
        Returns:
            LCA element or None
        """
        if not elements:  # If no elements provided
            return None  # Return None
        
        paths = [self._get_path_to_root(elem, parent_map) for elem in elements]  # Get path from each element to root
        
        min_length = min(len(path) for path in paths)  # Find shortest path length
        
        lca = None
        for i in range(min_length):  # Iterate through path positions
            if len(set(path[i] for path in paths)) == 1:  # If all paths have same element at position i
                lca = paths[0][i]  # This is the LCA (common ancestor)
            else:
                break  # Paths diverged, stop
        
        return lca  # Return the LCA found
    
    def _get_path_to_root(self, elem: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> List[ET.Element]:
        """Get path from element to root"""
        path = []  # Initialize empty path
        current = elem  # Start at element
        
        while current is not None:  # Traverse up to root
            path.append(current)  # Add current to path
            current = parent_map.get(current)  # Move to parent
        
        return list(reversed(path))  # Reverse path (root to element becomes element to root)

# ============================================================================
#  4.5: Active-Based Sibling Pruning (NEW!)
# ============================================================================

class ActiveBasedPruner:
    """Prunes stale subtrees by traversing up from active LCA"""
    
    def find_and_prune_stale_subtrees(
        self,
        root: ET.Element,
        active_elements: List[ET.Element],
        stale_elements: List[ET.Element],
        parent_map: Dict[ET.Element, ET.Element]
    ) -> int:
        """
        Main algorithm: Traverse from active LCA upward, prune stale siblings
        
        Args:
            root: XML root element
            active_elements: Elements visible in OCR
            stale_elements: Elements not visible in OCR
            parent_map: Parent mapping
            
        Returns:
            Number of elements removed
        """
        if not active_elements:
            return 0
        
        stale_set = set(stale_elements)
        lca_finder = LCAFinder()
        active_lca = lca_finder.find_lca(active_elements, parent_map)
        
        if active_lca is None:
            return 0
        
        total_removed = 0
        current = active_lca
        
        # Traverse upward from active LCA to root
        while current is not None:
            parent = parent_map.get(current)
            if parent is None:
                # Reached root, no more parents
                break
            
            # Get all siblings of current node (children of parent excluding current)
            siblings = [child for child in parent if child != current]
            
            # Check each sibling and prune if it contains stale elements
            for sibling in siblings:
                if self._subtree_contains_stale(sibling, stale_set):
                    removed = len(list(sibling.iter()))
                    parent.remove(sibling)
                    total_removed += removed
            
            # Move up to parent for next iteration
            current = parent
        
        return total_removed
    
    def _subtree_contains_stale(self, node: ET.Element, stale_set: Set[ET.Element]) -> bool:
        """Check if node or any descendant is stale"""
        for elem in node.iter():
            if elem in stale_set:
                return True
        return False

# ============================================================================
#  8: File Writer - Save cleaned XML
# ============================================================================

class XMLWriter:
    """Writes cleaned XML to file"""
    
    def save_cleaned_xml(self, tree: ET.ElementTree, output_path: str) -> None:
        """
        Save cleaned XML tree to file
        
        Args:
            tree: Cleaned XML tree
            output_path: Path to save cleaned XML
        """
        tree.write(output_path, encoding='utf-8', xml_declaration=True)  # Write XML to file with UTF-8 encoding and XML declaration

# ============================================================================
# PHASE 9: Orchestrator - Connects all phases (UPDATED)
# ============================================================================

class XMLCleaner:
    """Main orchestrator using active-based sibling pruning"""
    
    def __init__(self):
        self.loader = DataLoader()           # Phase 1
        self.ocr = OCRExtractor()            # Phase 2
        self.parser = XMLParser()            # Phase 3
        self.detector = StaleDetector()      # Phase 4 (UPDATED)
        self.pruner = ActiveBasedPruner()    # Phase 4.5 (NEW!)
        self.writer = XMLWriter()            # Phase 8
    
    def clean(self, image_path: str, xml_path: str, output_path: str) -> Dict:
        """
        Main workflow using active-based sibling pruning approach
        Simplified: Image + XML → Cleaned XML (for latency measurement)
        
        Args:
            image_path: Path to screenshot
            xml_path: Path to XML dump
            output_path: Path to save cleaned XML
            
        Returns:
            Statistics dict with total_latency and cleaning_latency
        """
        # Start timing for TOTAL end-to-end process
        total_start_time = time.time()
        
        # Phase 1: Load inputs
        img_path, tree = self.loader.load_inputs(image_path, xml_path)
        
        # Phase 2: OCR extraction (this is usually the bottleneck)
        ocr_start_time = time.time()
        visible_text = self.ocr.extract_visible_text(img_path)
        ocr_end_time = time.time()
        ocr_latency = ocr_end_time - ocr_start_time
        
        # Phase 3: Parse XML
        root, parent_map = self.parser.parse_xml(tree)
        total_elements = len(list(root.iter()))
        
        # Start timing for XML cleaning process (excluding OCR and file I/O)
        cleaning_start_time = time.time()
        
        # Phase 4: Find active AND stale elements
        active_elements, stale_elements = self.detector.find_active_and_stale(
            root, 
            visible_text
        )
        
        if not stale_elements:
            cleaning_end_time = time.time()
            cleaning_latency = cleaning_end_time - cleaning_start_time
            
            # End timing for total process
            total_end_time = time.time()
            total_latency = total_end_time - total_start_time
            
            return {
                'status': 'clean',
                'removed': 0,
                'visible_text': visible_text,
                'cleaning_latency': cleaning_latency,
                'ocr_latency': ocr_latency,
                'total_latency': total_latency
            }
        
        # Phase 4.5: Active-based sibling pruning
        removed = self.pruner.find_and_prune_stale_subtrees(
            root,
            active_elements,
            stale_elements,
            parent_map
        )
        
        # Phase 8: Save cleaned XML
        self.writer.save_cleaned_xml(tree, output_path)
        
        # End timing for XML cleaning process
        cleaning_end_time = time.time()
        cleaning_latency = cleaning_end_time - cleaning_start_time
        
        # End timing for total process
        total_end_time = time.time()
        total_latency = total_end_time - total_start_time
        
        return {
            'status': 'cleaned',
            'removed': removed,
            'method': 'active_based_sibling_pruning',
            'active_count': len(active_elements),
            'stale_count': len(stale_elements),
            'visible_text': visible_text,
            'cleaning_latency': cleaning_latency,
            'ocr_latency': ocr_latency,
            'total_latency': total_latency
        }

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Initialize cleaner
    cleaner = XMLCleaner()
    
    # File paths
    image_path = "/Users/suhas/Desktop/Projects/Clean_Xml/data/AtZXD_21.0_0_before_tap_Giczg.png"
    xml_path = "/Users/suhas/Desktop/Projects/Clean_Xml/data/Giczg_.xml"
    
    # Output directory
    output_dir = Path("/Users/suhas/Desktop/Projects/Clean_Xml/active_output")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Output path in active_output folder
    output_path = str(output_dir / "Giczg_cleaned.xml")
    
    print("="*60)
    print("XML Cleaner - Latency Measurement")
    print("="*60)
    print(f"Input Image: {image_path}")
    print(f"Input XML:   {xml_path}")
    print(f"Output XML:  {output_path}")
    print("\n🔄 Processing...")
    
    # Run cleaning
    result = cleaner.clean(image_path, xml_path, output_path)
    
    # Print results
    print("\n" + "="*60)
    print("✅ RESULTS")
    print("="*60)
    print(f"Status:           {result['status']}")
    print(f"Elements removed: {result.get('removed', 0)}")
    print(f"Active elements:  {result.get('active_count', 0)}")
    print(f"Stale elements:   {result.get('stale_count', 0)}")
    print("\n" + "-"*60)
    print("⏱️  LATENCY BREAKDOWN")
    print("-"*60)
    print(f"OCR Latency:           {result.get('ocr_latency', 0):.4f} seconds")
    print(f"XML Cleaning Latency:  {result.get('cleaning_latency', 0):.4f} seconds")
    print(f"TOTAL Latency:         {result.get('total_latency', 0):.4f} seconds")
    print("\n💡 For production reporting, use TOTAL Latency")
    print(f"\n💾 Cleaned XML saved to: {output_path}")
    print("="*60)
