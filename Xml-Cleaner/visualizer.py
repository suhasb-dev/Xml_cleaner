import matplotlib
matplotlib.use('Agg') # Non-interactive backend for web apps
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import xml.etree.ElementTree as ET
import cv2
import re
from typing import Dict, Set, Tuple

# ==========================================
# 1. Tree Visualizer (Refactored from your upload)
# ==========================================
class XMLTreeVisualizer:
    # Single color scheme for all nodes
    DEFAULT_COLOR = {'fill': '#E3F2FD', 'border': '#1976D2', 'text': '#000000'}
    HIGHLIGHT_COLOR = {'fill': '#FFF59D', 'border': '#F57F17', 'text': '#000000'}  # Yellow for active nodes
    
    def visualize(self, xml_path: str, output_path: str, visible_text: Set[str] = None, active_elements: Set = None):
        """Generates tree visualization. If visible_text and active_elements are provided, highlights active nodes."""
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Calculate Layout
        positions = self._calculate_layout(root)
        max_depth = max(p['level'] for p in positions.values())
        
        # Setup Figure with larger size for better text readability
        fig, ax = plt.subplots(figsize=(24, 18))
        ax.set_xlim(-len(positions)*0.5, len(positions)*0.5)
        ax.set_ylim(-max_depth * 2 - 2, 2)
        ax.axis('off')
        
        # Draw Edges
        self._draw_edges(ax, positions, root)
        
        # Draw Nodes
        self._draw_nodes(ax, positions, visible_text, active_elements)
        
        plt.title("XML Tree Structure" + (" (Active Nodes Highlighted)" if active_elements else ""), fontsize=16)
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)

    def _calculate_layout(self, root, x=0, y=0, level=0, spacing=2.5):
        positions = {}
        
        def _get_width(node):
            children = list(node)
            if not children: return 1.0
            return sum(_get_width(c) for c in children)

        def _assign_pos(node, curr_x, curr_y, curr_level):
            positions[id(node)] = {'x': curr_x, 'y': curr_y, 'level': curr_level, 'element': node}
            children = list(node)
            if not children: return
            
            width = sum(_get_width(c) for c in children)
            start_x = curr_x - (width * spacing / 2)
            
            current_offset = 0
            for child in children:
                child_w = _get_width(child)
                child_x = start_x + (current_offset + child_w/2) * spacing
                _assign_pos(child, child_x, curr_y - 2, curr_level + 1)
                current_offset += child_w
                
        _assign_pos(root, x, y, level)
        return positions

    def _draw_edges(self, ax, positions, node):
        node_id = id(node)
        if node_id not in positions: return
        parent_pos = positions[node_id]
        
        for child in node:
            child_id = id(child)
            if child_id in positions:
                child_pos = positions[child_id]
                arrow = FancyArrowPatch(
                    (parent_pos['x'], parent_pos['y']),
                    (child_pos['x'], child_pos['y']),
                    arrowstyle='-', color='#555', linewidth=1, zorder=1
                )
                ax.add_patch(arrow)
                self._draw_edges(ax, positions, child)

    def _draw_nodes(self, ax, positions, visible_text, active_elements):
        for pid, info in positions.items():
            elem = info['element']
            text = elem.get('text', '').strip()
            
            # Highlight Logic: Check if this element is in the active_elements set
            is_highlight = False
            if active_elements and elem in active_elements:
                is_highlight = True
            
            # Use single color scheme
            if is_highlight:
                face = self.HIGHLIGHT_COLOR['fill']
                edge = self.HIGHLIGHT_COLOR['border']
                lw = 3
            else:
                face = self.DEFAULT_COLOR['fill']
                edge = self.DEFAULT_COLOR['border']
                lw = 1
            
            # Calculate box size based on text length
            if text:
                # Use actual text, wrap if too long
                display_text = text
                # Wrap text if longer than 20 characters
                if len(display_text) > 20:
                    # Try to break at word boundaries
                    words = display_text.split()
                    lines = []
                    current_line = ""
                    for word in words:
                        if len(current_line + " " + word) <= 20:
                            current_line = current_line + " " + word if current_line else word
                        else:
                            if current_line:
                                lines.append(current_line)
                            current_line = word
                    if current_line:
                        lines.append(current_line)
                    display_text = "\n".join(lines[:2])  # Max 2 lines
                
                # Calculate box dimensions based on text
                num_lines = display_text.count('\n') + 1
                text_width = max(len(line) for line in display_text.split('\n')) if display_text else 0
                box_width = max(1.2, min(3.0, text_width * 0.15))
                box_height = max(0.8, 0.6 + (num_lines - 1) * 0.4)
            else:
                display_text = ""
                box_width = 1.2
                box_height = 0.8
            
            # Draw Box
            box = FancyBboxPatch(
                (info['x']-box_width/2, info['y']-box_height/2), box_width, box_height,
                boxstyle="round,pad=0.1",
                facecolor=face, edgecolor=edge, linewidth=lw, zorder=2
            )
            ax.add_patch(box)
            
            # Draw Text Label - only show actual text from XML, make it readable
            if display_text:
                # Use larger, readable font - adjust based on text length
                max_line_len = max(len(line) for line in display_text.split('\n')) if '\n' in display_text else len(display_text)
                if max_line_len <= 10:
                    fontsize = 11
                elif max_line_len <= 15:
                    fontsize = 10
                else:
                    fontsize = 9
                
                ax.text(info['x'], info['y'], display_text, 
                       ha='center', va='center', 
                       fontsize=fontsize, 
                       zorder=3,
                       family='sans-serif',
                       weight='normal')

# ==========================================
# 2. Bounding Box Visualizer (Refactored)
# ==========================================
class BoundingBoxVisualizer:
    COLORS = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]
    
    def visualize(self, image_path: str, xml_path: str, output_path: str):
        img = cv2.imread(image_path)
        if img is None: return
        
        tree = ET.parse(xml_path)
        idx = 0
        
        for elem in tree.getroot().iter():
            bounds = elem.get('bounds')
            if bounds:
                # Parse [x1,y1][x2,y2]
                match = re.match(r'\[(\d+),(\d+)\]\[(\d+),(\d+)\]', bounds)
                if match:
                    x1, y1, x2, y2 = map(int, match.groups())
                    color = self.COLORS[idx % len(self.COLORS)]
                    cv2.rectangle(img, (x1, y1), (x2, y2), color, 2)
                    idx += 1
        
        cv2.imwrite(output_path, img)