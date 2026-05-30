import fitz
import easyocr
import os
import sys

def main():
    # Fix console encoding issue for the progress bar characters
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass
        
    print("Initializing EasyOCR reader...")
    reader = easyocr.Reader(['en'], gpu=False)
    pdf_dir = 'Assignments'
    scanned_files = ['DL_LAB06.pdf', 'DL_LAB08.pdf', 'DL_LAB09.pdf', 'Exp 3 DL.pdf']
    
    with open('ocr_summary.txt', 'w', encoding='utf-8') as out_f:
        for name in scanned_files:
            path = os.path.join(pdf_dir, name)
            if not os.path.exists(path):
                out_f.write(f"File {name} does not exist.\n\n")
                continue
            
            out_f.write(f"==================================================\n")
            out_f.write(f"FILE: {name}\n")
            out_f.write(f"==================================================\n")
            print(f"Processing {name}...")
            try:
                doc = fitz.open(path)
                num_pages = len(doc)
                out_f.write(f"Total Pages: {num_pages}\n")
                
                # We will OCR page 0 (first page)
                page = doc[0]
                pix = page.get_pixmap(dpi=150)
                img_bytes = pix.tobytes('png')
                results = reader.readtext(img_bytes, detail=0)
                out_f.write("--- Page 1 OCR ---\n")
                out_f.write("\n".join(results[:30]))
                out_f.write("\n\n")
                
                # Let's also check if there is an middle page to OCR to get more context
                if num_pages > 2:
                    mid_page = doc[num_pages // 2]
                    pix = mid_page.get_pixmap(dpi=150)
                    img_bytes = pix.tobytes('png')
                    results_mid = reader.readtext(img_bytes, detail=0)
                    out_f.write(f"--- Page {num_pages // 2 + 1} OCR ---\n")
                    out_f.write("\n".join(results_mid[:30]))
                    out_f.write("\n\n")
                    
            except Exception as e:
                out_f.write(f"Error processing {name}: {str(e)}\n\n")
                print(f"Error processing {name}: {e}")
                
    print("OCR process completed. Output written to ocr_summary.txt")

if __name__ == '__main__':
    main()
