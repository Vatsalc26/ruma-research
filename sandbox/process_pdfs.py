import os
import glob
import shutil
import pymupdf4llm

def process_and_organize_pdfs():
    base_dir = r"d:\Prototypes\Project_Breaking"
    os.chdir(base_dir)
    
    # 1. Find all PDFs in the root directory
    pdf_files = [f for f in os.listdir(base_dir) if f.endswith('.pdf') and os.path.isfile(f)]
    
    if not pdf_files:
        print("No new PDF files found in the root directory to process.")
        return
        
    print(f"Found {len(pdf_files)} PDFs to process.")
    
    for pdf_file in pdf_files:
        base_name = os.path.splitext(pdf_file)[0]
        md_file = f"{base_name}.md"
        
        print(f"\n[Processing] {pdf_file}...")
        try:
            # 2. Convert PDF to Markdown using pymupdf4llm
            # By default this will extract text reliably without missing words
            md_text = pymupdf4llm.to_markdown(pdf_file, write_images=False)
            
            with open(md_file, "w", encoding="utf-8") as f:
                f.write(md_text)
            print(f"  -> Converted to Markdown: {md_file}")
            
            # 3. Organize: Create a directory named after the file
            target_dir = os.path.join(base_dir, base_name)
            if not os.path.exists(target_dir):
                os.makedirs(target_dir)
                
            # 4. Move files to the new directory
            shutil.move(pdf_file, os.path.join(target_dir, pdf_file))
            shutil.move(md_file, os.path.join(target_dir, md_file))
            print(f"  -> Organized into folder: {target_dir}")
            
        except Exception as e:
            print(f"  [ERROR] Failed to process {pdf_file}: {e}")

    # 5. Clean up any accidental .png files generated anywhere in the base directory
    print("\n[Cleanup] Looking for stray .png files to delete...")
    png_files = glob.glob(os.path.join(base_dir, "**", "*.png"), recursive=True)
    for png in png_files:
        try:
            os.remove(png)
            print(f"  -> Deleted image: {png}")
        except Exception as e:
            print(f"  [ERROR] Failed to delete {png}: {e}")
            
    print("\n[Done] All PDFs converted and organized.")

if __name__ == "__main__":
    process_and_organize_pdfs()
