"""

Removes stale screen elements from XML using Active-Based Sibling Pruning approach:
1. Find active elements (visible in OCR) and stale elements (invisible)
2. Find LCA of all active elements
3. Traverse upward from active LCA to root
4. At each level, check siblings and prune subtrees containing stale elements
"""

# Standard library
from pathlib import Path
from typing import List, Dict, Set, Tuple, Optional
import xml.etree.ElementTree as ET
import time

# Third-party (install: pip install easyocr)
import easyocr
import cv2
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
    
    def find_stale_elements(self, root: ET.Element, visible_text: Set[str]) -> List[ET.Element]:
        """Find elements not visible in image (legacy method for backward compatibility)"""
        _, stale_elements = self.find_active_and_stale(root, visible_text)
        return stale_elements
    
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
#  6: Validation - Check if safe to prune subtree (FIXED)
# ============================================================================

class PruningValidator:
    """Validates if LCA subtree can be safely removed"""
    
    def is_safe_to_prune(self, lca: ET.Element, root: ET.Element, visible_text: Set[str]) -> bool:
        """
        Check if removing LCA subtree still leaves all OCR text in remaining XML
        Uses SEMANTIC matching (same as StaleDetector)
        """
        # Filter OCR noise (same as detector)
        filtered_ocr = {t for t in visible_text if len(t) > 2}
        
        # Get all elements in LCA subtree
        stale_subtree = set(lca.iter())
        
        # Get remaining elements after removing LCA
        remaining_elements = [elem for elem in root.iter() if elem not in stale_subtree]
        
        # Extract text from remaining elements
        remaining_texts = {elem.get('text', '').lower().strip() 
                          for elem in remaining_elements 
                          if elem.get('text', '').strip() and len(elem.get('text', '').strip()) > 2}
        
        # Check if all meaningful OCR text can be found in remaining XML
        # Use semantic matching (same as detector)
        for ocr_text in filtered_ocr:
            if not self._is_text_present(ocr_text, remaining_texts):
                return False  # Would lose visible text
        
        return True  # Safe to prune
    
    def _is_text_present(self, ocr_text: str, remaining_texts: Set[str]) -> bool:
        """Check if OCR text is present using semantic matching"""
        ocr_tokens = set(re.findall(r'\w+', ocr_text))
        
        for xml_text in remaining_texts:
            xml_tokens = set(re.findall(r'\w+', xml_text))
            
            if ocr_tokens & xml_tokens:
                overlap_ratio = len(ocr_tokens & xml_tokens) / len(ocr_tokens)
                if overlap_ratio >= 0.5:  # Same threshold as detector
                    return True
        
        return False

# ============================================================================
# PHASE 7: Pruner - Execute removal of stale elements
# ============================================================================

class XMLPruner:
    """Removes stale elements from XML tree"""
    
    def prune_subtree(self, lca: ET.Element, parent_map: Dict[ET.Element, ET.Element]) -> int:
        """
        Remove entire LCA subtree
        
        Args:
            lca: LCA element to remove
            parent_map: Dict mapping child to parent
            
        Returns:
            Number of elements removed
        """
        count = len(list(lca.iter()))  # Count elements in subtree before removal
        
        parent = parent_map.get(lca)  # Get parent of LCA
        if parent is not None:  # If parent exists (LCA is not root)
            parent.remove(lca)  # Remove entire LCA subtree from parent
        
        return count  # Return count of removed elements
    
    def prune_individual(self, elements: List[ET.Element], parent_map: Dict[ET.Element, ET.Element]) -> int:
        """
        Remove individual elements (fallback when subtree pruning unsafe)
        
        Args:
            elements: List of elements to remove
            parent_map: Dict mapping child to parent
            
        Returns:
            Number of elements removed
        """
        count = 0  # Initialize counter
        
        for elem in elements:  # Iterate through elements to remove
            parent = parent_map.get(elem)  # Get parent of element
            if parent is not None:  # If parent exists
                parent.remove(elem)  # Remove element from parent
                count += 1  # Increment counter
        
        return count  # Return total removed

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
        
        Args:
            image_path: Path to screenshot
            xml_path: Path to XML dump
            output_path: Path to save cleaned XML
            
        Returns:
            Statistics dict
        """
        print("="*60)
        print("XML Cleaner - Active-Based Approach")
        print("="*60)
        
        # Phase 1: Load inputs
        print("\n📥 Phase 1: Loading inputs...")
        img_path, tree = self.loader.load_inputs(image_path, xml_path)
        
        # Phase 2: OCR extraction
        print("🔍 Phase 2: Running OCR...")
        visible_text = self.ocr.extract_visible_text(img_path)
        print(f"   Detected {len(visible_text)} visible texts")
        
        # Phase 3: Parse XML
        print("📊 Phase 3: Parsing XML...")
        root, parent_map = self.parser.parse_xml(tree)
        total_elements = len(list(root.iter()))
        print(f"   Total XML elements: {total_elements}")
        
        # Start timing for XML cleaning process
        cleaning_start_time = time.time()
        
        # Phase 4: Find active AND stale elements
        print("🔎 Phase 4: Detecting active & stale elements...")
        active_elements, stale_elements = self.detector.find_active_and_stale(
            root, 
            visible_text
        )
        
        if not stale_elements:
            cleaning_end_time = time.time()
            cleaning_latency = cleaning_end_time - cleaning_start_time
            print("✅ No stale elements found - XML is clean!")
            print(f"\n⏱️  XML Cleaning Latency: {cleaning_latency:.4f} seconds")
            return {
                'status': 'clean',
                'removed': 0,
                'visible_text': visible_text,
                'cleaning_latency': cleaning_latency
            }
        
        print(f"   Active elements: {len(active_elements)}")
        print(f"   Stale elements: {len(stale_elements)}")
        
        # Phase 4.5: Active-based sibling pruning
        print("\n🚀 Phase 4.5: Pruning stale subtrees...")
        removed = self.pruner.find_and_prune_stale_subtrees(
            root,
            active_elements,
            stale_elements,
            parent_map
        )
        
        print(f"   Removed {removed} elements via subtree pruning")
        
        # Phase 8: Save cleaned XML
        print("\n💾 Phase 8: Saving cleaned XML...")
        self.writer.save_cleaned_xml(tree, output_path)
        print(f"   Saved to: {output_path}")
        
        # End timing for XML cleaning process
        cleaning_end_time = time.time()
        cleaning_latency = cleaning_end_time - cleaning_start_time
        
        # Summary
        print("\n" + "="*60)
        print("✅ CLEANING COMPLETE")
        print("="*60)
        print(f"Original elements: {total_elements}")
        print(f"Active elements: {len(active_elements)}")
        print(f"Stale elements: {len(stale_elements)}")
        print(f"Removed: {removed}")
        print(f"Remaining: {total_elements - removed}")
        print(f"\n⏱️  XML Cleaning Latency: {cleaning_latency:.4f} seconds")
        print("="*60)
        
        return {
            'status': 'cleaned',
            'removed': removed,
            'method': 'active_based_sibling_pruning',
            'active_count': len(active_elements),
            'stale_count': len(stale_elements),
            'visible_text': visible_text,
            'cleaning_latency': cleaning_latency
        }


# ============================================================================
# Bounding Box Visualizer
# ============================================================================

class BoundingBoxVisualizer:
    """Draws bounding boxes from cleaned XML on the original image"""
    
    # Color palette: BGR format for OpenCV (Blue, Green, Red)
    COLORS = [
        (255, 0, 0),      # Blue
        (0, 255, 0),      # Green
        (0, 0, 255),      # Red
        (255, 255, 0),    # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255),   # Yellow
        (128, 0, 128),   # Purple
        (255, 165, 0),   # Orange
        (0, 128, 255),   # Light Blue
        (128, 255, 0),   # Lime
    ]
    
    def visualize(self, image_path: str, cleaned_xml_path: str, output_path: str) -> None:
        """
        Draw bounding boxes from cleaned XML on image
        
        Args:
            image_path: Path to original image
            cleaned_xml_path: Path to cleaned XML file
            output_path: Path to save visualization image
        """
        # Load image
        img = cv2.imread(image_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Parse XML and extract bounds
        tree = ET.parse(cleaned_xml_path)
        root = tree.getroot()
        
        # Draw bounding boxes for elements with bounds, using different colors
        color_index = 0
        for elem in root.iter():
            bounds = elem.get('bounds', '')
            if bounds:
                x1, y1, x2, y2 = self._parse_bounds(bounds)
                # Get color based on element properties
                color = self._get_color(elem, color_index)
                # Draw rectangle with different colors
                cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                color_index += 1
        
        # Save visualization
        cv2.imwrite(output_path, img)
    
    def _get_color(self, elem: ET.Element, index: int) -> Tuple[int, int, int]:
        """Get color for element based on its properties"""
        # Use different colors based on element attributes
        text = elem.get('text', '').strip()
        class_name = elem.get('class', '')
        
        if text:
            # Elements with text - use warm colors (red, orange, yellow)
            return self.COLORS[index % len(self.COLORS)]
        elif 'TextView' in class_name or 'Button' in class_name:
            # Text/Button elements - use blue/cyan
            return (255, 0, 0)  # Blue
        elif 'ImageView' in class_name:
            # Image elements - use green
            return (0, 255, 0)  # Green
        elif 'Layout' in class_name:
            # Layout containers - use purple
            return (128, 0, 128)  # Purple
        else:
            # Other elements - cycle through colors
            return self.COLORS[index % len(self.COLORS)]
    
    def _parse_bounds(self, bounds_str: str) -> Tuple[int, int, int, int]:
        """Parse bounds string '[x1,y1][x2,y2]' to coordinates"""
        match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds_str)
        if match:
            return tuple(map(int, match.groups()))
        return (0, 0, 0, 0)

# ============================================================================
# MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    # Initialize cleaner
    cleaner = XMLCleaner()
    
    # File paths from data folder
    image_path = "/Users/suhas/Desktop/Projects/Clean_Xml/data/AtZXD_21.0_0_before_tap_Giczg.png"
    xml_path = "/Users/suhas/Desktop/Projects/Clean_Xml/data/Giczg_.xml"
    output_path = "/Users/suhas/Desktop/Projects/Clean_Xml/data/Giczg_cleaned.xml"
    
    # Run cleaning (progress is printed by clean() method)
    result = cleaner.clean(image_path, xml_path, output_path)
    
    # Print results
    print(f"\n✅ Results:")
    print(f"  Status:        {result['status']}")
    print(f"  Elements removed: {result.get('removed', 0)}")
    print(f"  Method used:   {result.get('method', 'N/A')}")
    print(f"  Active found:  {result.get('active_count', 0)}")
    print(f"  Stale found:   {result.get('stale_count', 0)}")
    print(f"\n💾 Output saved to:")
    print(f"  {output_path}")
    
    # Display OCR visible text detection results
    visible_text = result.get('visible_text', set())
    if visible_text:
        print(f"\n📝 OCR Visible Text Detection Results:")
        print(f"  Total unique texts detected: {len(visible_text)}")
        print(f"\n  Detected texts:")
        # Sort for better readability
        sorted_texts = sorted(visible_text)
        for i, text in enumerate(sorted_texts, 1):
            print(f"    {i}. '{text}'")
        
        # Save OCR results to a text file
        ocr_output_file = output_path.replace('.xml', '_ocr_detected_texts.txt')
        try:
            with open(ocr_output_file, 'w', encoding='utf-8') as f:
                f.write("="*60 + "\n")
                f.write("OCR Visible Text Detection Results\n")
                f.write("="*60 + "\n\n")
                f.write(f"Total unique texts detected: {len(visible_text)}\n")
                f.write(f"Image: {image_path}\n")
                f.write(f"XML: {xml_path}\n")
                f.write(f"\n" + "-"*60 + "\n")
                f.write("Detected Texts (sorted alphabetically):\n")
                f.write("-"*60 + "\n\n")
                for i, text in enumerate(sorted_texts, 1):
                    f.write(f"{i}. '{text}'\n")
                f.write("\n" + "="*60 + "\n")
            print(f"\n💾 OCR results saved to:")
            print(f"  {ocr_output_file}")
        except Exception as e:
            print(f"\n⚠️  Warning: Could not save OCR results to file: {e}")
    else:
        print(f"\n⚠️  No visible text detected by OCR")
    
    # Visualize bounding boxes
    print(f"\n🎨 Creating bounding box visualization...")
    visualizer = BoundingBoxVisualizer()
    viz_output = "/Users/suhas/Desktop/Projects/Clean_Xml/data/Giczg_visualized.png"
    visualizer.visualize(image_path, output_path, viz_output)
    print(f"  Visualization saved to: {viz_output}")
    
    # Create tree visualization for original XML (before cleaning)
    print(f"\n🌳 Creating tree visualization for original XML (before cleaning)...")
    from xml_tree_visualizer import XMLTreeVisualizer
    tree_visualizer = XMLTreeVisualizer()
    before_tree_output = "/Users/suhas/Desktop/Projects/Clean_Xml/data/Giczg_before_tree.png"
    
    tree_visualizer.visualize_tree(
        xml_path=xml_path,
        output_path=before_tree_output,
        figsize=(60, 50),  # Increased figure size for better spacing
        show_attributes=False,  # Set to False to make nodes more compact
        compact_mode=True  # Use compact mode to reduce node sizes
    )
    print(f"  Original XML tree visualization saved to: {before_tree_output}")
    
    # Create tree visualization with OCR highlighting for cleaned XML
    print(f"\n🌳 Creating tree visualization with OCR highlighting (cleaned XML)...")
    tree_output = "/Users/suhas/Desktop/Projects/Clean_Xml/data/Giczg_tree_ocr_highlighted.png"
    
    # Get visible_text from results
    visible_text = result.get('visible_text', set())
    
    if visible_text:
        tree_visualizer.visualize_tree_with_ocr_highlight(
            xml_path=output_path,
            output_path=tree_output,
            visible_text=visible_text,
            figsize=(60, 50),  # Increased figure size for better spacing
            show_attributes=False,  # Set to False to make nodes more compact
            compact_mode=True  # Use compact mode to reduce node sizes
        )
        print(f"  Cleaned XML tree visualization saved to: {tree_output}")
    else:
        print(f"  ⚠️  No OCR text available for highlighting")
    
    print("\n" + "="*60)


