from __future__ import annotations

from pathlib import Path

import pdfplumber


PDF_PATH = Path(r"D:\soldercracks\Paper reference.pdf")


def main() -> None:
    if not PDF_PATH.exists():
        raise FileNotFoundError(PDF_PATH)

    with pdfplumber.open(PDF_PATH) as pdf:
        pages = pdf.pages[:5]
        text = "\n".join((p.extract_text() or "") for p in pages)
        out_path = PDF_PATH.with_suffix(".txt")
        alt_path = PDF_PATH.with_name("paper_reference.txt")
        out_path.write_text(text, encoding="utf-8")
        alt_path.write_text(text, encoding="utf-8")
        print(f"Wrote text to: {out_path}")
        print(f"Wrote text to: {alt_path}")


if __name__ == "__main__":
    main()
