"""
XML Tree Visualizer Web Application

A beautiful web-based UI for uploading and visualizing XML files as tree structures.
Simply upload your XML file and see the tree visualization instantly!

Author: Principal Software Engineer
"""

from flask import Flask, render_template, request, jsonify, send_file, redirect, url_for, session
import os
from pathlib import Path
import xml.etree.ElementTree as ET
from xml_tree_visualizer import XMLTreeVisualizer
import tempfile
import uuid
from datetime import datetime
import json

app = Flask(__name__)
app.secret_key = 'xml_visualizer_secret_key_change_in_production'

# Configuration
UPLOAD_FOLDER = Path('uploads')
OUTPUT_FOLDER = Path('outputs')
UPLOAD_FOLDER.mkdir(exist_ok=True)
OUTPUT_FOLDER.mkdir(exist_ok=True)

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['OUTPUT_FOLDER'] = OUTPUT_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Allowed extensions
ALLOWED_EXTENSIONS = {'xml'}

def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def generate_tree_structure(xml_path, node_id="1"):
    """Generate a JSON tree structure for interactive view with sequential numbering"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        
        # Counter for sequential node numbering
        node_counter = [1]  # Use list to allow modification in nested function
        
        def build_tree(element):
            """Recursively build tree structure with sequential numbering"""
            tag = element.tag.split('}')[-1]  # Remove namespace
            current_id = node_counter[0]
            node_counter[0] += 1  # Increment for next node
            
            # Get all attributes (not just key ones)
            attrs = dict(element.attrib)
            
            # Get text content
            text = element.text.strip() if element.text and element.text.strip() else None
            
            node = {
                'id': current_id,
                'tag': tag,
                'attributes': attrs,
                'text': text,
                'children': []
            }
            
            # Process children
            for child in element:
                node['children'].append(build_tree(child))
            
            return node
        
        return build_tree(root)
    except Exception as e:
        return {'error': str(e)}

def get_xml_stats(xml_path):
    """Get quick statistics about the XML file"""
    try:
        tree = ET.parse(xml_path)
        root = tree.getroot()
        all_elements = list(root.iter())
        
        # Count tags
        tag_counts = {}
        for elem in all_elements:
            tag = elem.tag.split('}')[-1]  # Remove namespace
            tag_counts[tag] = tag_counts.get(tag, 0) + 1
        
        # Count elements with text
        text_elements = sum(1 for elem in all_elements if elem.text and elem.text.strip())
        
        # Count elements with attributes
        attr_elements = sum(1 for elem in all_elements if elem.attrib)
        
        # Get max depth
        def get_depth(element, current_depth=0):
            if not list(element):
                return current_depth
            return max(get_depth(child, current_depth + 1) for child in element)
        
        max_depth = get_depth(root)
        
        # Top tags
        top_tags = sorted(tag_counts.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return {
            'success': True,
            'total_elements': len(all_elements),
            'text_elements': text_elements,
            'attr_elements': attr_elements,
            'unique_tags': len(tag_counts),
            'max_depth': max_depth,
            'top_tags': top_tags,
            'root_tag': root.tag.split('}')[-1]
        }
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and generate visualization"""
    try:
        # Check if file is present
        if 'file' not in request.files:
            return render_template('error.html', error='No file uploaded'), 400
        
        file = request.files['file']
        
        # Check if filename is empty
        if file.filename == '':
            return render_template('error.html', error='No file selected'), 400
        
        # Check if file is allowed
        if not allowed_file(file.filename):
            return render_template('error.html', error='Only XML files are allowed'), 400
        
        # Generate unique filename
        unique_id = str(uuid.uuid4())[:8]
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        
        # Save uploaded file
        xml_filename = f"{timestamp}_{unique_id}_{file.filename}"
        xml_path = app.config['UPLOAD_FOLDER'] / xml_filename
        file.save(str(xml_path))
        
        # Get visualization options from request
        # Checkboxes only send value when checked, so we check if they exist
        show_attributes = request.form.get('show_attributes') == 'true'
        compact_mode = request.form.get('compact_mode') == 'true'
        
        # Get XML statistics first
        stats = get_xml_stats(xml_path)
        
        if not stats['success']:
            return render_template('error.html', error=f"Invalid XML: {stats['error']}"), 400
        
        # Determine optimal figure size - REASONABLE dimensions for browser
        total_elements = stats['total_elements']
        max_depth = stats['max_depth']
        
        # Calculate reasonable size (browsers struggle with images > 10000x10000 pixels)
        # At 150 DPI: 40 inches = 6000 pixels (reasonable)
        if total_elements > 100:
            # Large tree: reasonable size
            figsize = (45, 35)
            compact_mode = False
        elif total_elements > 50:
            figsize = (35, 28)
            compact_mode = False
        else:
            figsize = (30, 22)
            compact_mode = False
        
        # Generate output filename
        output_filename = f"{timestamp}_{unique_id}_tree.png"
        output_path = app.config['OUTPUT_FOLDER'] / output_filename
        
        # Create visualizer and generate tree
        visualizer = XMLTreeVisualizer()
        visualizer.visualize_tree(
            xml_path=str(xml_path),
            output_path=str(output_path),
            figsize=figsize,
            show_attributes=show_attributes,
            compact_mode=compact_mode
        )
        
        # Store visualization data in session (without tree_structure to avoid cookie size limit)
        # We'll regenerate tree_structure from XML file when needed
        viz_id = f"{timestamp}_{unique_id}"
        session[f'viz_{viz_id}'] = {
            'image_path': f'/output/{output_filename}',
            'stats': stats,
            'filename': file.filename,
            'xml_filename': xml_filename  # Store XML filename to regenerate tree
        }
        
        # Redirect to visualization page
        return redirect(url_for('visualize', viz_id=viz_id))
        
    except ET.ParseError as e:
        return render_template('error.html', error=f'XML Parse Error: {str(e)}'), 400
    except Exception as e:
        return render_template('error.html', error=f'Error: {str(e)}'), 500

@app.route('/visualize/<viz_id>')
def visualize(viz_id):
    """Display the full-window visualization page"""
    # Retrieve visualization data from session
    viz_data = session.get(f'viz_{viz_id}')
    
    if not viz_data:
        return render_template('error.html', error='Visualization not found. Please upload a file again.'), 404
    
    # Regenerate tree structure from XML file (to avoid storing large data in session)
    xml_path = app.config['UPLOAD_FOLDER'] / viz_data['xml_filename']
    
    if not xml_path.exists():
        return render_template('error.html', error='XML file not found. Please upload again.'), 404
    
    try:
        tree_structure = generate_tree_structure(xml_path)
    except Exception as e:
        return render_template('error.html', error=f'Error generating tree structure: {str(e)}'), 500
    
    return render_template('visualize.html', 
                         viz_id=viz_id,
                         image_path=viz_data['image_path'],
                         stats=viz_data['stats'],
                         filename=viz_data['filename'],
                         tree_structure=json.dumps(tree_structure),
                         timestamp=int(datetime.now().timestamp()))

@app.route('/output/<filename>')
def serve_output(filename):
    """Serve generated visualization images with CORS headers"""
    file_path = app.config['OUTPUT_FOLDER'] / filename
    print(f"📂 Serving image: {file_path}")
    print(f"   File exists: {file_path.exists()}")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return jsonify({'error': 'File not found'}), 404
    
    file_size_mb = file_path.stat().st_size / (1024 * 1024)
    print(f"   File size: {file_size_mb:.2f} MB")
    
    if file_size_mb > 10:
        print(f"⚠️  WARNING: Large file size ({file_size_mb:.2f} MB) - may cause browser issues")
    
    response = send_file(
        file_path, 
        mimetype='image/png',
        as_attachment=False,
        download_name=filename,
        max_age=0  # Disable caching
    )
    
    # Add CORS headers
    response.headers['Access-Control-Allow-Origin'] = '*'
    response.headers['Cache-Control'] = 'no-cache, no-store, must-revalidate'
    response.headers['Pragma'] = 'no-cache'
    response.headers['Expires'] = '0'
    
    return response

@app.route('/cleanup', methods=['POST'])
def cleanup():
    """Clean up old files (optional endpoint)"""
    try:
        # Clean files older than 1 hour
        import time
        current_time = time.time()
        
        for folder in [app.config['UPLOAD_FOLDER'], app.config['OUTPUT_FOLDER']]:
            for file_path in folder.glob('*'):
                if file_path.is_file():
                    file_age = current_time - file_path.stat().st_mtime
                    if file_age > 3600:  # 1 hour
                        file_path.unlink()
        
        return jsonify({'success': True, 'message': 'Cleanup completed'})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})

if __name__ == '__main__':
    print("="*70)
    print("          🌳 XML TREE VISUALIZER - WEB UI 🌳")
    print("="*70)
    print()
    print("🚀 Starting web server...")
    print("📂 Upload folder:", UPLOAD_FOLDER.absolute())
    print("📂 Output folder:", OUTPUT_FOLDER.absolute())
    print()
    print("✨ Open your browser and go to:")
    print("   👉 http://localhost:5000")
    print()
    print("Press CTRL+C to stop the server")
    print("="*70)
    print()
    
    app.run(debug=True, host='0.0.0.0', port=5000)

