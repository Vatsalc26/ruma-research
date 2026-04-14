import os
import glob
import shutil
import fitz  # PyMuPDF core

def process_and_organize_pdfs_fast():
    base_dir = r"d:\Prototypes\Project_Breaking"
    os.chdir(base_dir)
    
    # 1. Find all PDFs in the root directory
    pdf_files = [f for f in os.listdir(base_dir) if f.endswith('.pdf') and os.path.isfile(f)]
    
    if not pdf_files:
        print("No new PDF files found in the root directory to process.")
        return
        
    print(f"Found {len(pdf_files)} PDFs to process via Fast Extraction.")
    
    for pdf_file in pdf_files:
        base_name = os.path.splitext(pdf_file)[0]
        md_file = f"{base_name}.md"
        
        print(f"\n[Processing] {pdf_file}...")
        try:
            # FAST EXTRACTION (No AI layout analyzing)
            doc = fitz.open(pdf_file)
            text_blocks = []
            for page in doc:
                text_blocks.append(page.get_text())
            doc.close()
            
            md_text = "\n\n".join(text_blocks)
            
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"  -> Extracted text: {md_file}")
            
            # Organize
            target_dir = os.path.join(base_dir, base_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                
            shutil.move(pdf_file, os.path.join(target_dir, pdf_file))
            shutil.move(md_file, os.path.join(target_dir, md_file))
            print(f"  -> Organized into folder: {target_dir}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {pdf_file}: {e}")

    # Aggressive Cleanup for any existing PNGs
    print("\n[Cleanup] Executing final purge of any .png files...")
    png_files = glob.glob(os.path.join(base_dir, "**", "*.png"), recursive=True)
    for png in png_files:
        try:
            os.remove(png)
        except:
            pass
            
    print("\n[Done] All fast extractions and organization complete.")

if __name__ == "__main__":
    process_and_organize_pdfs_fast()
