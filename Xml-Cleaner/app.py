import gradio as gr
import os
import tempfile
import shutil
from pathlib import Path
import time

# Import our modules
from ocr_strategies import OCRFactory
from core_cleaner import XMLCleanerCore
from visualizer import XMLTreeVisualizer, BoundingBoxVisualizer

# Initialize Logic Classes
cleaner_core = XMLCleanerCore()
tree_viz = XMLTreeVisualizer()
bbox_viz = BoundingBoxVisualizer()

def process_pipeline(image_file, xml_file, ocr_choice, visible_text_input, progress=gr.Progress()):
    # 1. Validation
    if xml_file is None:
        raise gr.Error("Please upload XML file.")
    
    # Check if we need image (only if visible text is not provided)
    use_ocr = not (visible_text_input and visible_text_input.strip())
    if use_ocr and image_file is None:
        raise gr.Error("Please upload Image file when using OCR, or provide visible text manually.")
        
    start_time = time.time()
    
    # 2. Setup Paths (Safe Temp Files)
    temp_dir = Path(tempfile.gettempdir())
    unique_id = str(int(time.time()))
    
    # Paths for outputs
    cleaned_xml_path = temp_dir / f"cleaned_{unique_id}.xml"
    
    img_viz_before = temp_dir / f"bbox_before_{unique_id}.png"
    img_viz_after = temp_dir / f"bbox_after_{unique_id}.png"
    tree_viz_before = temp_dir / f"tree_before_{unique_id}.png"
    tree_viz_after = temp_dir / f"tree_after_{unique_id}.png"
    
    # 3. Text Extraction Stage (OCR or Manual Input)
    text_source = None
    if visible_text_input and visible_text_input.strip():
        # Use provided visible text - NO OCR NEEDED
        progress(0.2, desc="Using provided visible text (OCR skipped)...")
        # Convert input text to set of strings (split by newlines or commas)
        lines = [line.strip() for line in visible_text_input.replace(',', '\n').split('\n') if line.strip()]
        visible_text = {line.lower().strip() for line in lines if line.strip()}
        text_source = "Manual Input"
    else:
        # Use OCR - image is required here
        progress(0.2, desc="Running OCR on image...")
        ocr_engine = OCRFactory.get_strategy(ocr_choice)
        visible_text = ocr_engine.extract_text(image_file)
        text_source = ocr_choice
    
    # 4. XML Parsing & Detection
    progress(0.4, desc="Parsing XML...")
    tree, root, parent_map = cleaner_core.parse_xml(xml_file)
    
    progress(0.5, desc="Detecting Stale Elements...")
    active, stale = cleaner_core.find_active_and_stale(root, visible_text)
    
    # 5. Pruning
    progress(0.6, desc="Pruning Tree...")
    removed_count = 0
    if stale:
        removed_count = cleaner_core.prune_stale_subtrees(root, active, stale, parent_map)
    
    # Save Cleaned XML
    tree.write(str(cleaned_xml_path))
    
    # 6. Visualization Generation
    progress(0.7, desc="Generating Visualizations...")
    
    # Bounding Boxes (only if image is provided)
    if image_file is not None:
        bbox_viz.visualize(image_file, xml_file, str(img_viz_before))
        bbox_viz.visualize(image_file, str(cleaned_xml_path), str(img_viz_after))
    else:
        # Create placeholder images or skip
        img_viz_before = None
        img_viz_after = None
    
    # Trees
    progress(0.8, desc="Drawing Trees (This might take a moment)...")
    # Before: no highlights
    tree_viz.visualize(xml_file, str(tree_viz_before), visible_text=None, active_elements=None)
    # After: highlight active elements (OCR matched nodes)
    active_elements_set = set(active) if active else set()
    tree_viz.visualize(str(cleaned_xml_path), str(tree_viz_after), visible_text, active_elements_set)
    
    # 7. Stats
    total_time = time.time() - start_time
    stats_md = f"""
    ### 📊 Process Statistics
    
    | Metric | Result |
    | :--- | :--- |
    | **Text Source** | {text_source} |
    | **Elements Removed** | `{removed_count}` |
    | **Active Elements** | `{len(active)}` |
    | **Stale Elements** | `{len(stale)}` |
    | **Processing Time** | `{total_time:.2f}s` |
    """
    
    ocr_text_display = "\n".join(sorted(list(visible_text)))
    
    progress(1.0, desc="Done!")
    
    return (
        str(tree_viz_before),
        str(tree_viz_after),
        str(img_viz_before) if img_viz_before else None,
        str(img_viz_after) if img_viz_after else None,
        stats_md,
        ocr_text_display,
        str(cleaned_xml_path)
    )

# --- Gradio UI Layout ---
custom_css = """
.container { max-width: 1100px; margin: auto; }
.header { text-align: center; margin-bottom: 20px; }
.stat-box { border: 1px solid #ddd; padding: 10px; border-radius: 8px; background: #f9f9f9; }
"""

with gr.Blocks() as app:
    
    with gr.Row():
        gr.Markdown(
            """
            # 🌳 XML Cleaner & Visualizer Studio
            **Optimize Mobile UI XMLs** by removing invisible/stale nodes using OCR-based or manual text input for sibling pruning.
            """, 
            elem_classes="header"
        )
        
    with gr.Row():
        # --- Left Panel: Inputs ---
        with gr.Column(scale=1, variant="panel"):
            gr.Markdown("### 1. Upload Data")
            img_input = gr.Image(type="filepath", label="Screenshot (PNG/JPG)")
            gr.Markdown("*Optional if visible text is provided below*")
            xml_input = gr.File(label="XML Layout Dump", file_types=[".xml"])
            
            gr.Markdown("### 2. Visible Text (Optional)")
            visible_text_input = gr.TextArea(
                label="Visible Text",
                placeholder="Enter visible text from the screenshot (one per line or comma-separated). Leave empty to use OCR.",
                lines=5,
                info="If provided, this text will be used instead of OCR. Otherwise, OCR will be used automatically."
            )
            
            # Status indicator for text input mode
            text_input_status = gr.Markdown("", visible=False)
            
            gr.Markdown("### 3. Settings")
            ocr_selector = gr.Dropdown(
                choices=["EasyOCR (Best Accuracy)", "Tesseract (Fast & Free)"],
                value="EasyOCR (Best Accuracy)",
                label="OCR Engine (Fallback)",
                info="Used only if visible text is not provided above.",
                interactive=True
            )
            
            btn_run = gr.Button("✨ Run Analysis & Clean", variant="primary", size="lg")

        # --- Right Panel: Outputs ---
        with gr.Column(scale=2):
            gr.Markdown("### 4. Analysis Results")
            
            # Stats Area
            stats_output = gr.Markdown()
            
            # Visualization Tabs
            with gr.Tabs():
                with gr.TabItem("🌳 Tree Structure"):
                    gr.Markdown("*Left: Original XML | Right: Cleaned XML (Active Nodes Highlighted)*")
                    with gr.Row():
                        out_tree_before = gr.Image(label="Before Pruning", type="filepath")
                        out_tree_after = gr.Image(label="After Pruning", type="filepath")
                
                with gr.TabItem("🖼️ Bounding Boxes"):
                    gr.Markdown("*Visualizing XML bounds on the screenshot*")
                    with gr.Row():
                        out_bbox_before = gr.Image(label="Original Bounds", type="filepath")
                        out_bbox_after = gr.Image(label="Cleaned Bounds", type="filepath")
                
                with gr.TabItem("📝 OCR Data"):
                    out_ocr_text = gr.TextArea(label="Detected Text", lines=10, interactive=False)
            
            # Download
            gr.Markdown("### 5. Export")
            out_file = gr.File(label="Download Cleaned XML")

    # Function to toggle OCR selector and image input based on visible text input
    def toggle_ocr_selector(visible_text):
        """Disable OCR selector if visible text is provided, enable if empty"""
        if visible_text and visible_text.strip():
            return (
                gr.update(
                    label="OCR Engine (Disabled - Using Manual Text)",
                    info="⚠️ OCR is disabled because visible text is provided above.",
                    interactive=False
                ),
                gr.update(value="✅ **Using Manual Text Input** - OCR is disabled. Image is optional.", visible=True),
                gr.update(label="Screenshot (PNG/JPG) - Optional")
            )
        else:
            return (
                gr.update(
                    label="OCR Engine",
                    info="Select OCR engine to extract visible text from the screenshot.",
                    interactive=True
                ),
                gr.update(value="", visible=False),
                gr.update(label="Screenshot (PNG/JPG) - Required")
            )
    
    # Wire Interactions
    # Update OCR selector and image input when visible text changes
    visible_text_input.change(
        fn=toggle_ocr_selector,
        inputs=[visible_text_input],
        outputs=[ocr_selector, text_input_status, img_input]
    )
    
    btn_run.click(
        fn=process_pipeline,
        inputs=[img_input, xml_input, ocr_selector, visible_text_input],
        outputs=[
            out_tree_before, out_tree_after, 
            out_bbox_before, out_bbox_after, 
            stats_output, out_ocr_text, out_file
        ]
    )

if __name__ == "__main__":
    app.launch(css=custom_css, theme=gr.themes.Soft())