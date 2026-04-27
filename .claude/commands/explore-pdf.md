Analyze a CVM filing PDF to understand its structure. The user will provide a path to a PDF file as $ARGUMENTS.

Steps:
1. Open the PDF with PyMuPDF (fitz)
2. Report: total pages, file size
3. For each page (first 10 max), extract text blocks and report:
   - Page number
   - Number of text blocks
   - Whether any tables are detected (via pdfplumber)
   - Heading-like text (larger font size or bold)
4. Attempt section detection using the regex patterns from `src/parsing/section_detector.py` (if it exists). Report which sections were found and on which pages.
5. Extract and display one sample table (if any found) to show the raw table structure
6. Note any parsing anomalies: empty pages, scanned-image pages (no text extracted), unusual encodings

This is a diagnostic tool — output should be informative, not pretty. Focus on revealing the PDF's structure so Rafael can calibrate parsing rules.
