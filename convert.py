import pypandoc
import sys

print("Downloading pandoc...")
try:
    pypandoc.download_pandoc()
except Exception as e:
    print(f"Pandoc download failed: {e}")

print("Converting to DOCX...")
try:
    pypandoc.convert_file('paper_draft.md', 'docx', outputfile='paper.docx')
    print("DOCX created successfully.")
except Exception as e:
    print(f"DOCX failed: {e}")

print("Converting to TEX...")
try:
    pypandoc.convert_file('paper_draft.md', 'tex', outputfile='paper.tex')
    print("TEX created successfully.")
except Exception as e:
    print(f"TEX failed: {e}")
