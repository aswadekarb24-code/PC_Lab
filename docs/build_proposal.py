import docx
from docx.shared import Inches, Pt
from docx.enum.text import WD_ALIGN_PARAGRAPH

def create_proposal_document():
    doc = docx.Document()
    
    # Page Margins
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1)
        section.bottom_margin = Inches(1)
        section.left_margin = Inches(1)
        section.right_margin = Inches(1)

    # --- 1. TITLE ---
    title_p = doc.add_paragraph()
    title_p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    title_run = title_p.add_run("Hybrid Parallel Acceleration for Content-Aware Image Sequencing and High-FPS Smooth Video Transition Synthesis")
    title_run.bold = True
    title_run.font.size = Pt(16)
    title_run.font.name = 'Calibri'

    doc.add_paragraph() # Spacer

    # --- 2. TEAM DETAILS ---
    h2 = doc.add_heading("2. Team Details", level=2)
    
    table = doc.add_table(rows=3, cols=3)
    table.style = 'Table Grid'
    
    # Header Row
    hdr_cells = table.rows[0].cells
    hdr_cells[0].text = "Role / Member"
    hdr_cells[1].text = "Name"
    hdr_cells[2].text = "Roll No / Lab Batch"
    
    for cell in hdr_cells:
        for paragraph in cell.paragraphs:
            for run in paragraph.runs:
                run.bold = True

    # Member 1 Row
    row1_cells = table.rows[1].cells
    row1_cells[0].text = "Member 1"
    row1_cells[1].text = "Awwab Wadekar"
    row1_cells[2].text = "241080077 | C"

    # Member 2 Row
    row2_cells = table.rows[2].cells
    row2_cells[0].text = "Member 2"
    row2_cells[1].text = "Taha Valiji"
    row2_cells[2].text = "241080071 | C"

    doc.add_paragraph() # Spacer

    # --- 3. INTRODUCTION ---
    doc.add_heading("3. Introduction", level=2)
    intro_p = doc.add_paragraph(
        "Converting large, unordered collections of digital images into smooth, continuous video streams presents significant "
        "computational bottlenecks. Specifically, calculating pairwise visual feature distances across thousands of images requires "
        "O(N^2) complexity, and rendering pixel-level transition frames at high frame rates (60+ FPS) exceeds standard CPU processing "
        "capabilities. This project leverages parallel computing paradigms—namely OpenMP for shared-memory multi-core CPU preprocessing, "
        "MPI/OpenMP for structural distance matrix computations, and CUDA for hardware-accelerated GPU pixel interpolation—to achieve "
        "real-time video synthesis within hardware constraints."
    )
    intro_p.style.font.name = 'Calibri'

    # --- 4. GOAL ---
    doc.add_heading("4. Goal", level=2)
    goal_p = doc.add_paragraph(
        "The primary goal of this project is to build an efficient C++ parallel pipeline that:\n"
        "1. Ingests and processes a dataset of 1,000+ unstructured 720p images using multi-threaded OpenMP CPU operations.\n"
        "2. Computes a structural feature similarity matrix and solves a Nearest Neighbor (TSP) sorting heuristic to order images for smooth visual transitions.\n"
        "3. Implements custom CUDA kernels on an NVIDIA GPU to accelerate fine-grained pixel alpha blending and frame interpolation.\n"
        "4. Directly pipes generated frames into a hardware-accelerated NVENC video encoder to output a 60 FPS, 3-minute video without disk I/O bottlenecks."
    )
    goal_p.style.font.name = 'Calibri'

    # --- 5. REFERENCES (RESEARCH PAPERS) ---
    doc.add_heading("5. References (Research Papers)", level=2)
    ref1 = doc.add_paragraph(style='List Bullet')
    ref1.add_run("Snavely, N., Seitz, S. M., & Szeliski, R. ").bold = True
    ref1.add_run("\"Photo Tourism: Exploring Photo Collections in 3D.\" ACM Transactions on Graphics (SIGGRAPH) — utilized for structured image dataset feature representation and pairwise matching concepts.")

    ref2 = doc.add_paragraph(style='List Bullet')
    ref2.add_run("Niklaus, S., Mai, L., & Liu, F. ").bold = True
    ref2.add_run("\"Video Frame Interpolation via Adaptive Separable Convolution.\" IEEE International Conference on Computer Vision (ICCV) — referenced for GPU-accelerated intermediate frame generation principles.")

    # --- 6. EXPECTED RESULTS ---
    doc.add_heading("6. Expected Results", level=2)
    res_p = doc.add_paragraph(
        "By utilizing OpenMP multi-threading, similarity matrix computation time is expected to decrease by approximately 75% "
        "compared to single-threaded baseline execution. Using CUDA streaming on an NVIDIA RTX 4050 (4GB VRAM), the system will "
        "successfully synthesize over 10,000 intermediate transition frames, maintaining a continuous generation rate exceeding 60 FPS "
        "and producing a seamless, high-framerate output video."
    )
    res_p.style.font.name = 'Calibri'

    # Save document
    filename = "Project_Proposal_Abstract.docx"
    doc.save(filename)
    print(f"[SUCCESS] Proposal document successfully generated: {filename}")

if __name__ == "__main__":
    create_proposal_document()