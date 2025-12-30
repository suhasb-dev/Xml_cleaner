"""
Refactored XML Cleaner with dependency injection.
"""

import time
from pathlib import Path
from typing import Dict, List, Tuple, Set, Optional
from logging import getLogger
import xml.etree.ElementTree as ET
import asyncio
import re
from datasketch import MinHash

from ocr_strategies import OCRStrategy
from functools import wraps

logger = getLogger(__name__)


# Simple decorators for compatibility (can be enhanced later)
def profile_it(func_name: str = "", tags: dict = None):
    """Simple profiling decorator placeholder"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator


def json_log():
    """Simple logging decorator placeholder"""
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            return await func(*args, **kwargs)
        return wrapper
    return decorator


# Adapter to make OCRStrategy async-compatible
class BaseOCR:
    """Adapter to make OCRStrategy work with async interface"""
    
    def __init__(self, ocr_strategy: OCRStrategy):
        self._strategy = ocr_strategy
    
    async def extract_visible_text(self, image_path: Path) -> Set[str]:
        """Extract visible text asynchronously"""
        def _extract():
            return self._strategy.extract_text(str(image_path))
        return await asyncio.to_thread(_extract)


class DataLoader:
    """Loads and validates image and XML inputs asynchronously"""
    
    async def load_inputs(self, image_path: str, xml_path: str) -> Tuple[Path, ET.ElementTree]:
        img_path = Path(image_path)
        xml_file = Path(xml_path)
        
        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        if not xml_file.exists():
            raise FileNotFoundError(f"XML not found: {xml_path}")
        
        def _parse_xml():
            return ET.parse(xml_file)
        
        tree = await asyncio.to_thread(_parse_xml)
        return img_path, tree


class XMLParser:
    """Parses XML and builds parent-child relationships"""
    
    def parse_xml(self, tree: ET.ElementTree) -> Tuple[ET.Element, Dict[ET.Element, ET.Element]]:
        root = tree.getroot()
        parent_map = {child: parent for parent in root.iter() for child in parent}
        return root, parent_map


class StaleDetector:
    """
    Identifies active and stale elements by comparing XML with OCR results.
    
    Logic:
    - Active elements: XML elements whose text matches OCR results (visible on screen)
    - Stale elements: XML elements whose text does NOT match OCR results (not visible)
    """
    
    def find_active_and_stale(
        self,
        root: ET.Element,
        visible_text: Set[str]
    ) -> Tuple[List[ET.Element], List[ET.Element]]:
        """
        Compare all XML elements with OCR results to classify as active or stale.
        
        Args:
            root: Root element of the XML tree to process
            visible_text: Set of text strings extracted from OCR (visible on screen)
     
        Returns:
            Tuple containing:
                - List of active elements (text matches OCR results)
                - List of stale elements (text does not match OCR results)
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
        """
        Check if text matches any visible text using MinHash-based Jaccard similarity.
        
        Uses MinHash algorithm for efficient similarity comparison:
        1. Tokenizes element text and OCR text into word tokens
        2. Creates MinHash signatures for both texts
        3. Calculates Jaccard similarity between MinHashes
        4. Returns True if similarity >= 50%
        
        MinHash provides better performance and accuracy for text similarity
        compared to simple token overlap, especially for longer texts.
        
        Args:
            elem_text: Text from XML element
            visible_text: Set of text strings from OCR results
            
        Returns:
            True if element text semantically matches any OCR text, False otherwise
        """
        # Tokenize element text into word tokens
        elem_tokens = re.findall(r'\w+', elem_text.lower())
        
        # Create MinHash for element text
        if not elem_tokens:
            return False
        
        elem_minhash = MinHash(num_perm=32)
        for token in elem_tokens:
            elem_minhash.update(token.encode())
        
        # Compare with each OCR text result using MinHash
        for ocr_text in visible_text:
            # Tokenize OCR text into word tokens
            ocr_tokens = re.findall(r'\w+', ocr_text.lower())
            
            if not ocr_tokens:
                continue
            
            # Create MinHash for OCR text
            ocr_minhash = MinHash(num_perm=32)
            for token in ocr_tokens:
                ocr_minhash.update(token.encode())
            
            # Calculate Jaccard similarity using MinHash
            similarity = elem_minhash.jaccard(ocr_minhash)
            
            # Match if similarity >= 50%
            if similarity >= 0.5:
                return True
        
        return False


class LCAFinder:
    """Finds lowest common ancestor of elements"""
    
    def find_lca(
        self, 
        elements: List[ET.Element], 
        parent_map: Dict[ET.Element, ET.Element]
    ) -> Optional[ET.Element]:
        """
        Find the lowest common ancestor (LCA) of a list of XML elements.
        
        Traverses up from each element to the root, finding the deepest node
        that is an ancestor of all given elements.
        
        Args:
            elements: List of XML elements to find LCA for
            parent_map: Dictionary mapping each element to its parent
            
        Returns:
            The lowest common ancestor element, or None if elements list is empty
        """
        if not elements:
            return None
        
        paths = [self._get_path_to_root(elem, parent_map) for elem in elements]
        min_length = min(len(path) for path in paths)
        
        lca = None
        for i in range(min_length):
            if len(set(path[i] for path in paths)) == 1:
                lca = paths[0][i]
            else:
                break
        
        return lca
    
    def _get_path_to_root(
        self, 
        elem: ET.Element, 
        parent_map: Dict[ET.Element, ET.Element]
    ) -> List[ET.Element]:
        path = []
        current = elem
        
        while current is not None:
            path.append(current)
            current = parent_map.get(current)
        
        return list(reversed(path))


class ActiveBasedPruner:
    """Prunes stale subtrees by traversing up from active LCA"""
    
    def find_and_prune_stale_subtrees(
        self,
        root: ET.Element,
        active_elements: List[ET.Element],
        stale_elements: List[ET.Element],
        parent_map: Dict[ET.Element, ET.Element]
    ) -> int:
        if not active_elements:
            return 0
        
        stale_set = set(stale_elements)
        lca_finder = LCAFinder()
        active_lca = lca_finder.find_lca(active_elements, parent_map)
        
        if active_lca is None:
            return 0
        
        total_removed = 0
        current = active_lca
        
        while current is not None:
            parent = parent_map.get(current)
            if parent is None:
                break
            
            siblings = [child for child in parent if child != current]
            
            for sibling in siblings:
                if self._subtree_contains_stale(sibling, stale_set):
                    removed = len(list(sibling.iter()))
                    parent.remove(sibling)
                    total_removed += removed
            
            current = parent
        
        return total_removed
    
    def _subtree_contains_stale(self, node: ET.Element, stale_set: Set[ET.Element]) -> bool:
        for elem in node.iter():
            if elem in stale_set:
                return True
        return False


class XMLWriter:
    """Writes cleaned XML to file asynchronously"""
    
    async def save_cleaned_xml(self, tree: ET.ElementTree, output_path: str) -> None:
        def _write_xml():
            tree.write(output_path, encoding='utf-8', xml_declaration=True)
        
        await asyncio.to_thread(_write_xml)


class XMLCleaner:
    """
    XML Cleaner with injected OCR dependency.
    Now testable and flexible!
    """
    
    def __init__(
        self,
        ocr: BaseOCR,
        thread_code: str = ""
    ):
        """
        Args:
            ocr: OCR provider (BaseOCR adapter wrapping OCRStrategy)
            thread_code: Thread code for logging
        """
        self._ocr = ocr
        self._thread_code = thread_code
        
        # Create instances of supporting classes
        self._loader = DataLoader()
        self._parser = XMLParser()
        self._detector = StaleDetector()
        self._pruner = ActiveBasedPruner()
        self._writer = XMLWriter()
    
    @json_log()
    @profile_it(
        func_name="xml_cleaner", 
        tags={"operation_type": "xml_processing", "workflow": ""}
    )
    async def clean(
        self,
        image_path: str,
        xml_path: str,
        output_path: str
    ) -> Dict:
        """
        Main workflow - simplified interface.
        
        Args:
            image_path: Path to screenshot
            xml_path: Path to XML dump
            output_path: Path to save cleaned XML
            
        Returns:
            Statistics dict with detailed latency breakdown
        """
        total_start_time = time.perf_counter()
        
        try:
      
            load_start = time.perf_counter()
            img_path, tree = await self._loader.load_inputs(image_path, xml_path)
            load_latency = time.perf_counter() - load_start
            
            
            ocr_start = time.perf_counter()
            visible_text = await self._ocr.extract_visible_text(img_path)
            ocr_latency = time.perf_counter() - ocr_start
            
     
            parse_start = time.perf_counter()
            root, parent_map = self._parser.parse_xml(tree)
            total_elements = len(list(root.iter()))
            parse_latency = time.perf_counter() - parse_start
            
         
            detect_start = time.perf_counter()
            active_elements, stale_elements = self._detector.find_active_and_stale(
                root, visible_text
            )
            detect_latency = time.perf_counter() - detect_start
            
            # Early exit if no stale elements
            if not stale_elements:
                total_latency = time.perf_counter() - total_start_time
                return {
                    'status': 'clean',
                    'removed': 0,
                    'total_elements': total_elements,
                    'active_count': len(active_elements),
                    'visible_text_count': len(visible_text),
                    'load_latency': load_latency,
                    'ocr_latency': ocr_latency,
                    'parse_latency': parse_latency,
                    'detection_latency': detect_latency,
                    'total_latency': total_latency
                }
            
     
            prune_start = time.perf_counter()
            removed = self._pruner.find_and_prune_stale_subtrees(
                root, active_elements, stale_elements, parent_map
            )
            prune_latency = time.perf_counter() - prune_start
            
          
            save_start = time.perf_counter()
            await self._writer.save_cleaned_xml(tree, output_path)
            save_latency = time.perf_counter() - save_start
            
            total_latency = time.perf_counter() - total_start_time
            
            logger.info(
                f"XML cleaning completed: removed {removed}/{total_elements} elements "
                f"in {total_latency:.2f}s"
            )
            
            return {
                'status': 'cleaned',
                'removed': removed,
                'total_elements': total_elements,
                'method': 'active_based_sibling_pruning',
                'active_count': len(active_elements),
                'stale_count': len(stale_elements),
                'visible_text_count': len(visible_text),
                'load_latency': load_latency,
                'ocr_latency': ocr_latency,
                'parse_latency': parse_latency,
                'detection_latency': detect_latency,
                'pruning_latency': prune_latency,
                'save_latency': save_latency,
                'total_latency': total_latency
            }
            
        except Exception as e:
            logger.error(f"Error in XML cleaner: {e}", exc_info=True)
            total_latency = time.perf_counter() - total_start_time
            return {
                'status': 'error',
                'error': str(e),
                'total_latency': total_latency
            }


# Backward compatibility: Keep the old class name
class XMLCleanerCore:
    def __init__(self):
        pass # Stateless, pure logic

    def parse_xml(self, xml_path):
        tree = ET.parse(xml_path)
        root = tree.getroot()
        parent_map = {c: p for p in root.iter() for c in p}
        return tree, root, parent_map

    def find_active_and_stale(self, root, visible_text: Set[str]):
        active_elements = []
        stale_elements = []
        
        # Filter noise from OCR
        clean_ocr = {t for t in visible_text if len(t) > 2}
        
        for elem in root.iter():
            text = elem.get('text', '').lower().strip()
            if text and len(text) > 2:
                if self._is_semantic_match(text, clean_ocr):
                    active_elements.append(elem)
                else:
                    stale_elements.append(elem)
        return active_elements, stale_elements

    def _is_semantic_match(self, elem_text, visible_texts):
        # Token based matching
        elem_tokens = set(re.findall(r'\w+', elem_text))
        for ocr in visible_texts:
            ocr_tokens = set(re.findall(r'\w+', ocr))
            if not elem_tokens: continue
            overlap = len(elem_tokens & ocr_tokens)
            if overlap / len(elem_tokens) >= 0.5: # 50% match
                return True
        return False

    def prune_stale_subtrees(self, root, active_elements, stale_elements, parent_map):
        if not active_elements: return 0
        
        # 1. Find LCA of active elements
        active_lca = self._find_lca(active_elements, parent_map)
        if not active_lca: return 0
        
        stale_set = set(stale_elements)
        removed_count = 0
        current = active_lca
        
        # 2. Traverse Up and Prune Siblings
        while current is not None:
            parent = parent_map.get(current)
            if not parent: break
            
            siblings = [child for child in parent if child != current]
            for sibling in siblings:
                # If sibling tree has stale elements?
                # Simplified: If sibling is strictly in stale list or contains them
                if self._subtree_has_stale(sibling, stale_set):
                    removed_count += len(list(sibling.iter()))
                    parent.remove(sibling)
            
            current = parent
        return removed_count

    def _subtree_has_stale(self, node, stale_set):
        for x in node.iter():
            if x in stale_set: return True
        return False

    def _find_lca(self, elements, parent_map):
        # Get paths to root
        paths = []
        for el in elements:
            path = []
            curr = el
            while curr:
                path.append(curr)
                curr = parent_map.get(curr)
            paths.append(list(reversed(path)))
            
        if not paths: return None
        
        # Find common prefix
        min_len = min(len(p) for p in paths)
        lca = None
        for i in range(min_len):
            if len(set(p[i] for p in paths)) == 1:
                lca = paths[0][i]
            else:
                break
        return lca