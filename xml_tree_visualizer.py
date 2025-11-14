"""
XML Tree Visualizer

A comprehensive tool to visualize XML file structures as interactive tree diagrams.
Upload any XML file and get a beautiful tree visualization showing the hierarchy,
attributes, and relationships between elements.

Author: Principal Software Engineer
"""

import xml.etree.ElementTree as ET
from lxml import etree as lxml_etree
from pathlib import Path
from typing import Optional, Dict, List, Tuple, Set
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend (fixes threading issues)
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import textwrap


class XMLTreeVisualizer:
    """
    Professional XML Tree Visualizer that creates beautiful hierarchical 
    tree diagrams from XML files.
    """
    
    # Color scheme for professional visualization - depth-based colors (darker for better visibility)
    DEPTH_COLORS = [
        {'fill': '#BBDEFB', 'border': '#0D47A1', 'text': '#000000'},  # Level 0 - Darker Blue
        {'fill': '#E1BEE7', 'border': '#4A148C', 'text': '#000000'},  # Level 1 - Darker Purple
        {'fill': '#C8E6C9', 'border': '#1B5E20', 'text': '#000000'},  # Level 2 - Darker Green
        {'fill': '#FFE0B2', 'border': '#BF360C', 'text': '#000000'},  # Level 3 - Darker Orange
        {'fill': '#F8BBD0', 'border': '#880E4F', 'text': '#000000'},  # Level 4 - Darker Pink
        {'fill': '#B2DFDB', 'border': '#004D40', 'text': '#000000'},  # Level 5 - Darker Teal
        {'fill': '#DCEDC8', 'border': '#33691E', 'text': '#000000'},  # Level 6 - Darker Light Green
        {'fill': '#FFF9C4', 'border': '#F57F17', 'text': '#000000'},  # Level 7 - Darker Amber
    ]
    
    COLORS = {
        'line_color': '#424242',  # Darker gray for better visibility
        'background': '#FAFAFA',
    }
    
    def __init__(self):
        """Initialize the XML Tree Visualizer"""
        self.node_positions = {}
        self.node_counter = 0
        self.max_depth = 0
        
    def load_xml(self, xml_path: str) -> ET.ElementTree:
        """
        Load and parse XML file
        
        Args:
            xml_path: Path to XML file
            
        Returns:
            Parsed XML tree
            
        Raises:
            FileNotFoundError: If XML file doesn't exist
            ET.ParseError: If XML is malformed
        """
        xml_file = Path(xml_path)
        
        if not xml_file.exists():
            raise FileNotFoundError(f"❌ XML file not found: {xml_path}")
        
        try:
            tree = ET.parse(xml_file)
            print(f"✅ Successfully loaded XML: {xml_file.name}")
            return tree
        except ET.ParseError as e:
            raise ET.ParseError(f"❌ Invalid XML format: {str(e)}")
    
    def load_xml_lxml(self, xml_path: str) -> lxml_etree._ElementTree:
        """
        Load and parse XML file using lxml (for xpath calculation)
        
        Args:
            xml_path: Path to XML file
            
        Returns:
            Parsed lxml tree
            
        Raises:
            FileNotFoundError: If XML file doesn't exist
            lxml_etree.XMLSyntaxError: If XML is malformed
        """
        xml_file = Path(xml_path)
        
        if not xml_file.exists():
            raise FileNotFoundError(f"❌ XML file not found: {xml_path}")
        
        try:
            tree = lxml_etree.parse(str(xml_file))
            return tree
        except lxml_etree.XMLSyntaxError as e:
            raise lxml_etree.XMLSyntaxError(f"❌ Invalid XML format: {str(e)}")
    
    def get_element_info(self, element: ET.Element) -> Dict:
        """
        Extract comprehensive information from an XML element
        
        Args:
            element: XML element
            
        Returns:
            Dictionary with element information
        """
        info = {
            'tag': element.tag,
            'attributes': dict(element.attrib),
            'text': element.text.strip() if element.text and element.text.strip() else None,
            'children_count': len(list(element)),
            'tail': element.tail.strip() if element.tail and element.tail.strip() else None
        }
        return info
    
    def _create_element_mapping(self, et_root: ET.Element, lxml_root: lxml_etree._Element) -> Dict[ET.Element, lxml_etree._Element]:
        """
        Create a mapping between ET.Element and lxml_etree._Element by matching tags and positions
        
        Args:
            et_root: Root element from xml.etree.ElementTree
            lxml_root: Root element from lxml
            
        Returns:
            Dictionary mapping ET.Element to lxml_etree._Element
        """
        mapping = {}
        
        def match_elements(et_elem, lxml_elem):
            """Recursively match elements by tag and position"""
            mapping[et_elem] = lxml_elem
            
            # Match children by tag and order
            et_children = list(et_elem)
            lxml_children = list(lxml_elem)
            
            # Group by tag
            et_by_tag = {}
            for child in et_children:
                tag = child.tag.split('}')[-1]  # Remove namespace
                if tag not in et_by_tag:
                    et_by_tag[tag] = []
                et_by_tag[tag].append(child)
            
            lxml_by_tag = {}
            for child in lxml_children:
                tag = child.tag.split('}')[-1]  # Remove namespace
                if tag not in lxml_by_tag:
                    lxml_by_tag[tag] = []
                lxml_by_tag[tag].append(child)
            
            # Match children with same tags
            for tag in et_by_tag:
                if tag in lxml_by_tag:
                    et_list = et_by_tag[tag]
                    lxml_list = lxml_by_tag[tag]
                    for i, et_child in enumerate(et_list):
                        if i < len(lxml_list):
                            match_elements(et_child, lxml_list[i])
        
        match_elements(et_root, lxml_root)
        return mapping
    
    def get_xpath(self, element: ET.Element, lxml_tree: lxml_etree._ElementTree, 
                  element_to_lxml: Dict[ET.Element, lxml_etree._Element]) -> str:
        """
        Calculate XPath for an XML element using lxml
        
        Args:
            element: XML element to get xpath for (ET.Element)
            lxml_tree: Parsed lxml tree
            element_to_lxml: Dictionary mapping ET.Element to lxml_etree._Element
            
        Returns:
            XPath string for the element
        """
        try:
            # Get corresponding lxml element
            lxml_element = element_to_lxml.get(element)
            if lxml_element is not None:
                # Use lxml's getpath method
                return lxml_tree.getpath(lxml_element)
            else:
                # Fallback: return root path
                return "/"
        except Exception as e:
            # Fallback if xpath calculation fails
            print(f"⚠️  Warning: Could not calculate xpath: {e}")
            return "/"
    
    def _has_text(self, element: ET.Element) -> bool:
        """Check if element has non-empty text attribute"""
        text = element.get('text', '').strip()
        return bool(text)
    
    def calculate_tree_layout(self, root: ET.Element, x: float = 0, y: float = 0, 
                             level: int = 0, x_spacing: float = 2.0, 
                             y_spacing: float = 1.5) -> Dict:
        """
        Calculate positions for all nodes using improved algorithm that accounts for subtree widths
        Shows ALL nodes (both with and without text)
        
        Args:
            root: Root XML element
            x: Initial x position
            y: Initial y position
            level: Current depth level
            x_spacing: Horizontal spacing between nodes
            y_spacing: Vertical spacing between levels
            
        Returns:
            Dictionary mapping elements to their (x, y) positions
        """
        positions = {}
        subtree_widths = {}
        
        def _calculate_subtree_width(element):
            """Calculate the width needed for each subtree (bottom-up) - includes all nodes"""
            children = list(element)
            if not children:
                # Leaf node needs minimal width
                subtree_widths[id(element)] = x_spacing
                return x_spacing
            
            # Calculate width needed for all children's subtrees
            total_width = 0
            for child in children:
                total_width += _calculate_subtree_width(child)
            
            # Add spacing between subtrees
            if len(children) > 1:
                total_width += (len(children) - 1) * x_spacing
            
            subtree_widths[id(element)] = total_width
            return total_width
        
        def _calculate_positions(element, x, y, level, node_id="1"):
            """Recursive helper to calculate positions for all nodes"""
            nonlocal positions
            
            # Store position for ALL elements
            element_id = id(element)
            positions[element_id] = {
                'x': x, 
                'y': y, 
                'level': level, 
                'element': element,
                'node_id': node_id
            }
            
            # Track max depth
            if level > self.max_depth:
                self.max_depth = level
            
            # Calculate positions for children
            children = list(element)
            if children:
                # Get total width needed for all children
                total_width = subtree_widths[element_id]
                
                # Start positioning children from left
                current_x = x - (total_width / 2)
                
                for i, child in enumerate(children):
                    child_id = id(child)
                    child_subtree_width = subtree_widths[child_id]
                    
                    # Position child at center of its subtree
                    child_x = current_x + (child_subtree_width / 2)
                    child_y = y - y_spacing
                    
                    # Create hierarchical ID
                    child_node_id = f"{node_id}.{i+1}"
                    
                    # Recursively position child and its descendants
                    _calculate_positions(child, child_x, child_y, level + 1, child_node_id)
                    
                    # Move to next child's starting position
                    current_x += child_subtree_width + x_spacing
        
        # First pass: calculate subtree widths (bottom-up)
        _calculate_subtree_width(root)
        
        # Second pass: calculate positions (top-down)
        _calculate_positions(root, x, y, level, "1")
        
        return positions
    
    def create_node_label(self, element: ET.Element, max_length: int = 30) -> str:
        """
        Create a formatted label for a node
        
        Args:
            element: XML element
            max_length: Maximum length for text truncation
            
        Returns:
            Formatted label string
        """
        info = self.get_element_info(element)
        
        # Start with tag name
        label = f"{info['tag']}"
        
        # Add important attributes (limit to 2-3 key ones)
        important_attrs = ['id', 'class', 'name', 'type', 'resource-id', 'text']
        attr_lines = []
        
        for attr in important_attrs:
            if attr in info['attributes']:
                value = str(info['attributes'][attr])
                if len(value) > max_length:
                    value = value[:max_length] + "..."
                attr_lines.append(f"{attr}='{value}'")
                if len(attr_lines) >= 2:  # Limit to 2 attributes
                    break
        
        if attr_lines:
            label += "\n" + "\n".join(attr_lines)
        
        # Add text content if exists
        if info['text'] and len(info['text']) > 0:
            text = info['text']
            if len(text) > max_length:
                text = text[:max_length] + "..."
            label += f"\n📝: {text}"
        
        return label
    
    def visualize_tree(self, xml_path: str, output_path: str, 
                       figsize: Tuple[int, int] = (20, 12),
                       show_attributes: bool = True,
                       compact_mode: bool = False) -> None:
        """
        Create and save a visual tree diagram of the XML structure
        
        Args:
            xml_path: Path to input XML file
            output_path: Path to save the visualization
            figsize: Figure size (width, height)
            show_attributes: Whether to show attributes in nodes
            compact_mode: Use compact layout for large trees
        """
        # Load XML with both ET and lxml
        tree = self.load_xml(xml_path)
        root = tree.getroot()
        
        # Load with lxml for xpath calculation
        lxml_tree = self.load_xml_lxml(xml_path)
        lxml_root = lxml_tree.getroot()
        
        # Create mapping between ET and lxml elements
        element_to_lxml = self._create_element_mapping(root, lxml_root)
        
        # Calculate layout
        print("🔄 Calculating tree layout...")
        self.max_depth = 0
        
        # Adjust spacing based on tree complexity - INCREASED for larger nodes
        children_count = len(list(root.iter()))
        if children_count > 200:
            x_spacing = 18.0  # Much wider spacing for larger nodes
            y_spacing = 12.0   # Much taller spacing
        elif children_count > 100:
            x_spacing = 15.0
            y_spacing = 10.0
        elif children_count > 50:
            x_spacing = 12.0
            y_spacing = 8.0
        else:
            x_spacing = 10.0
            y_spacing = 7.0
        
        positions = self.calculate_tree_layout(root, x=0, y=0, 
                                              x_spacing=x_spacing, 
                                              y_spacing=y_spacing)
        
        print(f"📊 Tree statistics:")
        print(f"   Total nodes: {len(positions)}")
        print(f"   Max depth: {self.max_depth}")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.COLORS['background'])
        ax.set_facecolor(self.COLORS['background'])
        
        # Remove axes - MUCH MORE EXPANDED limits to accommodate wider tree
        # Calculate approximate tree width based on max depth and spacing
        max_tree_width = children_count * x_spacing * 0.5  # Estimate based on node count
        ax.set_xlim(-max_tree_width, max_tree_width)
        ax.set_ylim(-self.max_depth * y_spacing - 3, 3)
        ax.axis('off')
        
        # Draw edges first (so they appear behind nodes)
        print("🎨 Drawing tree structure...")
        self._draw_edges(ax, positions, root)
        
        # Draw nodes with xpath, text, and text_found_in_ocr (all False for regular tree)
        visible_text = set()  # Empty set for regular tree (no OCR matching)
        self._draw_nodes(ax, positions, root, lxml_tree, element_to_lxml, visible_text, show_attributes, compact_mode)
        
        # Add title
        xml_name = Path(xml_path).name
        plt.title(f"XML Tree Structure: {xml_name}", 
                 fontsize=18, fontweight='bold', 
                 color='#1565C0', pad=20)
        
        # Add legend
        self._add_legend(ax, children_count)
        
        # Save figure - Use 150 DPI for reasonable file size and browser compatibility
        # 150 DPI: 45 inches = 6750 pixels (browsers can handle this)
        # 300 DPI: 45 inches = 13500 pixels (too large for most browsers!)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor=self.COLORS['background'])
        plt.close(fig)  # Close figure to free memory
        
        # Print file info for debugging
        import os
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"💾 Visualization saved to: {output_path}")
        print(f"   File size: {file_size:.2f} MB")
        
        # Show statistics
        self._print_statistics(root)
    
    def visualize_tree_with_ocr_highlight(self, xml_path: str, output_path: str,
                                          visible_text: Set[str],
                                          figsize: Tuple[int, int] = (20, 12),
                                          show_attributes: bool = True,
                                          compact_mode: bool = False) -> None:
        """
        Create and save a visual tree diagram with OCR text highlighting.
        All nodes use the same color, and nodes matching OCR text are highlighted.
        
        Args:
            xml_path: Path to input XML file
            output_path: Path to save the visualization
            visible_text: Set of visible text strings from OCR (lowercased)
            figsize: Figure size (width, height)
            show_attributes: Whether to show attributes in nodes
            compact_mode: Use compact layout for large trees
        """
        # Load XML with both ET and lxml
        tree = self.load_xml(xml_path)
        root = tree.getroot()
        
        # Load with lxml for xpath calculation
        lxml_tree = self.load_xml_lxml(xml_path)
        lxml_root = lxml_tree.getroot()
        
        # Create mapping between ET and lxml elements
        element_to_lxml = self._create_element_mapping(root, lxml_root)
        
        # Calculate layout
        print("🔄 Calculating tree layout...")
        self.max_depth = 0
        
        # Adjust spacing based on tree complexity - INCREASED for larger nodes
        children_count = len(list(root.iter()))
        if children_count > 200:
            x_spacing = 18.0  # Much wider spacing for larger nodes
            y_spacing = 12.0   # Much taller spacing
        elif children_count > 100:
            x_spacing = 15.0
            y_spacing = 10.0
        elif children_count > 50:
            x_spacing = 12.0
            y_spacing = 8.0
        else:
            x_spacing = 10.0
            y_spacing = 7.0
        
        positions = self.calculate_tree_layout(root, x=0, y=0, 
                                              x_spacing=x_spacing, 
                                              y_spacing=y_spacing)
        
        print(f"📊 Tree statistics:")
        print(f"   Total nodes: {len(positions)}")
        print(f"   Max depth: {self.max_depth}")
        print(f"   OCR text matches: {len(visible_text)} unique texts")
        
        # Create figure
        fig, ax = plt.subplots(figsize=figsize, facecolor=self.COLORS['background'])
        ax.set_facecolor(self.COLORS['background'])
        
        # Remove axes - MUCH MORE EXPANDED limits to accommodate wider tree
        # Calculate approximate tree width based on max depth and spacing
        max_tree_width = children_count * x_spacing * 0.5  # Estimate based on node count
        ax.set_xlim(-max_tree_width, max_tree_width)
        ax.set_ylim(-self.max_depth * y_spacing - 3, 3)
        ax.axis('off')
        
        # Draw edges first (so they appear behind nodes)
        print("🎨 Drawing tree structure with OCR highlighting...")
        self._draw_edges(ax, positions, root)
        
        # Draw nodes with OCR highlighting
        self._draw_nodes_with_ocr_highlight(ax, positions, visible_text, root, lxml_tree, element_to_lxml, show_attributes, compact_mode)
        
        # Add title
        xml_name = Path(xml_path).name
        plt.title(f"XML Tree Structure (OCR Highlighted): {xml_name}", 
                 fontsize=18, fontweight='bold', 
                 color='#1565C0', pad=20)
        
        # Add legend with OCR info
        self._add_ocr_legend(ax, children_count, visible_text)
        
        # Save figure
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight', 
                   facecolor=self.COLORS['background'])
        plt.close(fig)  # Close figure to free memory
        
        # Print file info for debugging
        import os
        file_size = os.path.getsize(output_path) / (1024 * 1024)  # MB
        print(f"💾 Visualization saved to: {output_path}")
        print(f"   File size: {file_size:.2f} MB")
        
        # Show statistics
        self._print_statistics(root)
    
    def _draw_edges(self, ax, positions: Dict, root: ET.Element) -> None:
        """Draw edges (connections) between parent and child nodes - for all nodes"""
        def draw_connections(element):
            element_id = id(element)
            if element_id not in positions:
                return
            
            parent_pos = positions[element_id]
            
            for child in element:
                child_id = id(child)
                if child_id in positions:
                    child_pos = positions[child_id]
                    
                    # Draw arrow from parent to child - much thicker lines for better visibility
                    arrow = FancyArrowPatch(
                        (parent_pos['x'], parent_pos['y']),
                        (child_pos['x'], child_pos['y']),
                        arrowstyle='-',
                        color=self.COLORS['line_color'],
                        linewidth=5.0,
                        alpha=0.9,
                        zorder=1
                    )
                    ax.add_patch(arrow)
                    
                    # Recursively draw connections for children
                    draw_connections(child)
        
        draw_connections(root)
    
    def _draw_nodes(self, ax, positions: Dict, root: ET.Element, 
                   lxml_tree: lxml_etree._ElementTree, element_to_lxml: Dict[ET.Element, lxml_etree._Element],
                   visible_text: Set[str], show_attributes: bool, 
                   compact_mode: bool) -> None:
        """
        Draw nodes (rectangles/circles) for each XML element.
        - Nodes with empty text: just a box/circle with no label
        - Nodes with text: box/circle with only the text displayed
        """
        for element_id, pos_info in positions.items():
            element = pos_info['element']
            x, y = pos_info['x'], pos_info['y']
            level = pos_info['level']
            
            # Get color scheme based on depth level
            color_scheme = self.DEPTH_COLORS[level % len(self.DEPTH_COLORS)]
            
            # Get element text
            elem_text = element.get('text', '').strip()
            
            # Determine box size - larger sizes for better visibility
            if elem_text:
                # Node with text - size based on text length (larger sizes)
                max_text_length = 25 if compact_mode else 35
                text_display = elem_text[:max_text_length] + "..." if len(elem_text) > max_text_length else elem_text
                # Increased base size and scaling factor
                box_width = max(len(text_display) * 0.15 + 1.0, 2.0)
                box_height = 0.8
                label = text_display
            else:
                # Empty node - larger box for visibility
                box_width = 1.2
                box_height = 1.2
                label = None
            
            # Draw box/circle with much thicker borders and darker colors
            # Use darker fill color for better visibility
            darker_fill = color_scheme['fill']
            darker_border = color_scheme['border']
            # Make fill slightly darker by reducing lightness
            box = FancyBboxPatch(
                (x - box_width/2, y - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.2",
                facecolor=darker_fill,
                edgecolor=darker_border,
                linewidth=6.0,  # Much thicker border
                zorder=2,
                alpha=1.0  # Fully opaque
            )
            ax.add_patch(box)
            
            # Add text label only if node has text (much larger font)
            if label:
                text_fontsize = 16 if compact_mode else 18
                ax.text(x, y, label,
                       ha='center', va='center',
                       fontsize=text_fontsize,
                       color=color_scheme['text'],
                       weight='bold',
                       zorder=3)
    
    def _draw_nodes_with_ocr_highlight(self, ax, positions: Dict, visible_text: Set[str],
                                       root: ET.Element, lxml_tree: lxml_etree._ElementTree,
                                       element_to_lxml: Dict[ET.Element, lxml_etree._Element],
                                       show_attributes: bool, compact_mode: bool) -> None:
        """
        Draw nodes with same color for all, highlighting nodes that match OCR text.
        - Nodes with empty text: just a box/circle with no label
        - Nodes with text: box/circle with only the text displayed (highlighted if matches OCR)
        
        Args:
            ax: Matplotlib axes
            positions: Dictionary mapping element IDs to position info
            visible_text: Set of visible text from OCR (lowercased)
            root: Root element of the XML tree
            lxml_tree: Parsed lxml tree
            element_to_lxml: Dictionary mapping ET.Element to lxml_etree._Element
            show_attributes: Whether to show attributes
            compact_mode: Use compact layout
        """
        # Single color scheme for all nodes (not depth-based) - darker for better visibility
        NORMAL_COLOR = {'fill': '#BBDEFB', 'border': '#0D47A1', 'text': '#000000'}  # Darker blue
        HIGHLIGHT_COLOR = {'fill': '#FFE082', 'border': '#E65100', 'text': '#000000'}  # Darker Yellow/Orange for OCR matches
        
        for element_id, pos_info in positions.items():
            element = pos_info['element']
            x, y = pos_info['x'], pos_info['y']
            
            # Get element text
            elem_text = element.get('text', '').strip()
            elem_text_lower = elem_text.lower().strip() if elem_text else ''
            
            # Check if this element's text matches OCR text
            text_found_in_ocr = bool(elem_text_lower and elem_text_lower in visible_text)
            
            # Use highlight color if OCR match, otherwise normal color
            color_scheme = HIGHLIGHT_COLOR if text_found_in_ocr else NORMAL_COLOR
            
            # Determine box size - larger sizes for better visibility
            if elem_text:
                # Node with text - size based on text length (larger sizes)
                max_text_length = 25 if compact_mode else 35
                text_display = elem_text[:max_text_length] + "..." if len(elem_text) > max_text_length else elem_text
                # Increased base size and scaling factor
                box_width = max(len(text_display) * 0.15 + 1.0, 2.0)
                box_height = 0.8
                label = text_display
            else:
                # Empty node - larger box for visibility
                box_width = 1.2
                box_height = 1.2
                label = None
            
            # Draw box - much thicker border for OCR matches and better visibility
            border_width = 7.0 if text_found_in_ocr else 6.0
            box = FancyBboxPatch(
                (x - box_width/2, y - box_height/2),
                box_width, box_height,
                boxstyle="round,pad=0.2",
                facecolor=color_scheme['fill'],
                edgecolor=color_scheme['border'],
                linewidth=border_width,
                zorder=2,
                alpha=1.0  # Fully opaque
            )
            ax.add_patch(box)
            
            # Add OCR indicator badge for highlighted nodes (larger)
            if text_found_in_ocr and label:
                ocr_badge_x = x + box_width/2 - 0.2
                ocr_badge_y = y - box_height/2 + 0.2
                ocr_badge_fontsize = 16 if compact_mode else 18
                ax.text(ocr_badge_x, ocr_badge_y, "✓",
                       ha='center', va='center',
                       fontsize=ocr_badge_fontsize,
                       color='white',
                       weight='bold',
                       zorder=5,
                       bbox=dict(boxstyle='round,pad=0.15', 
                               facecolor='#4CAF50',  # Green checkmark
                               edgecolor='none',
                               alpha=1.0))
            
            # Add text label only if node has text (much larger font)
            if label:
                text_fontsize = 16 if compact_mode else 18
                ax.text(x, y, label,
                       ha='center', va='center',
                       fontsize=text_fontsize,
                       color=color_scheme['text'],
                       weight='bold',
                       zorder=3)
    
    def _add_ocr_legend(self, ax, node_count: int, visible_text: Set[str]) -> None:
        """Add legend with OCR highlighting information"""
        legend_text = (f"📊 Total Elements: {node_count}\n"
                      f"📏 Tree Depth: {self.max_depth} levels\n"
                      f"🎨 All nodes: Same color (Light Blue)\n"
                      f"✨ OCR Matches: Highlighted (Yellow/Orange)\n"
                      f"   • Thicker border + Green ✓ badge\n"
                      f"   • Matches {len(visible_text)} OCR text strings\n"
                      f"ℹ️  Empty nodes: box only\n"
                      f"   Nodes with text: text displayed")
                      # Node ID notation removed - commented out:
                      # f"🔢 [ID] = Node Notation\n"
                      # f"   • [1] = Root\n"
                      # f"   • [1.2] = 2nd child of [1]\n"
                      # f"   • [1.2.3] = 3rd child of [1.2]")
        ax.text(0.02, 0.98, legend_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor='white',
                        edgecolor='#1565C0',
                        linewidth=2,
                        alpha=0.95))
    
    def _add_legend(self, ax, node_count: int) -> None:
        """Add legend with information"""
        legend_text = (f"📊 Total Elements: {node_count}\n"
                      f"📏 Tree Depth: {self.max_depth} levels\n"
                      f"🎨 Color = Depth Level\n"
                      f"ℹ️  Empty nodes: box only\n"
                      f"   Nodes with text: text displayed")
                      # Node ID notation removed - commented out:
                      # f"🔢 [ID] = Node Notation\n"
                      # f"   • [1] = Root\n"
                      # f"   • [1.2] = 2nd child of [1]\n"
                      # f"   • [1.2.3] = 3rd child of [1.2]")
        ax.text(0.02, 0.98, legend_text,
               transform=ax.transAxes,
               fontsize=10,
               verticalalignment='top',
               fontweight='bold',
               bbox=dict(boxstyle='round,pad=0.8', 
                        facecolor='white',
                        edgecolor='#1565C0',
                        linewidth=2,
                        alpha=0.95))
    
    def _print_statistics(self, root: ET.Element) -> None:
        """Print detailed statistics about the XML structure"""
        all_elements = list(root.iter())
        elements_with_text = [elem for elem in all_elements if self._has_text(elem)]
        
        # Count tags
        tag_counts = {}
        for elem in all_elements:
            tag = elem.tag.split('}')[-1]  # Remove namespace
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Count elements with text
        text_elements = len(elements_with_text)
        
        # Count elements with attributes
        attr_elements = sum(1 for elem in all_elements if elem.attrib)
        
        print("\n" + "="*60)
        print("📈 XML STRUCTURE STATISTICS")
        print("="*60)
        print(f"Total Elements: {len(all_elements)}")
        print(f"Elements with Text: {text_elements}")
        print(f"Elements without Text: {len(all_elements) - text_elements}")
        print(f"Elements with Attributes: {attr_elements}")
        print(f"Unique Tag Types: {len(tag_counts)}")
        print(f"\n🏷️  Top 5 Most Common Tags:")
        for tag, count in sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]:
            print(f"   {tag}: {count}")
        print("="*60 + "\n")
    
    def create_interactive_html(self, xml_path: str, output_html: str) -> None:
        """
        Create an interactive HTML visualization (future enhancement)
        
        Args:
            xml_path: Path to XML file
            output_html: Path to save HTML file
        """
        # Placeholder for future D3.js or interactive visualization
        print("⚠️  Interactive HTML visualization - Coming soon!")
        print("    Current version generates static PNG/SVG images.")


def main():
    """
    Main entry point for the XML Tree Visualizer
    Demonstrates usage with example files
    """
    print("="*70)
    print("          🌳 XML TREE VISUALIZER 🌳")
    print("          Professional XML Structure Visualization")
    print("="*70)
    print()
    
    # Initialize visualizer
    visualizer = XMLTreeVisualizer()
    
    # Example usage with the existing XML file
    xml_path = "/Users/suhas/Desktop/Projects/Clean_Xml/data/Giczg_.xml"
    output_path = "/Users/suhas/Desktop/Projects/Clean_Xml/data/Giczg_tree_visualization.png"
    
    print(f"📂 Input XML: {xml_path}")
    print(f"💾 Output Image: {output_path}")
    print()
    
    try:
        # Create visualization
        visualizer.visualize_tree(
            xml_path=xml_path,
            output_path=output_path,
            figsize=(24, 16),
            show_attributes=True,
            compact_mode=False  # Set to True for very large XML files
        )
        
        print()
        print("✨ Visualization completed successfully!")
        print(f"📸 Open the file to view your XML tree: {output_path}")
        
    except FileNotFoundError as e:
        print(f"❌ Error: {str(e)}")
    except ET.ParseError as e:
        print(f"❌ XML Parse Error: {str(e)}")
    except Exception as e:
        print(f"❌ Unexpected error: {str(e)}")
    
    print()
    print("="*70)
    print("HOW TO USE WITH YOUR OWN XML FILES:")
    print("="*70)
    print("""
    from xml_tree_visualizer import XMLTreeVisualizer
    
    # Create visualizer instance
    visualizer = XMLTreeVisualizer()
    
    # Visualize your XML file
    visualizer.visualize_tree(
        xml_path='path/to/your/file.xml',
        output_path='path/to/output/tree.png',
        figsize=(20, 12),           # Adjust size as needed
        show_attributes=True,        # Show element attributes
        compact_mode=False           # Use compact layout for large files
    )
    """)
    print("="*70)


if __name__ == "__main__":
    main()

