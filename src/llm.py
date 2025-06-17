import pymupdf4llm
import re
import os
from pathlib import Path


def output_directory(filename):
    # Create a directory based on the filename without extension
    base_filename = os.path.splitext(os.path.basename(filename))[0]
    out_dir = f"md/{base_filename}"
    os.makedirs(out_dir, exist_ok=True)
    return out_dir


def filter_sections(sections, filename):
    if "_PMP_" in filename.name.upper():
        try:
            intro_idx = sections.index("**INTRODUCTION**")
            return sections[intro_idx - 1 :]
        except ValueError:
            pass
    return sections


def is_outletter(filename):
    # Check if the filename contains "OUTLETTER" in uppercase
    return "_OUTLETTER_" in filename.name.upper()


def pdf_to_markdown(filename):
    md_text = pymupdf4llm.to_markdown(filename)
    # Write the full markdown text to a single file
    out_dir = output_directory(filename)
    out_filename = os.path.join(out_dir, "full.md")
    with open(out_filename, "w", encoding="utf-8") as f:
        f.write(md_text)

    if is_outletter(filename):
        return  # they are just one section. Bail here.

    # Split the markdown text into sections based on bold caps headings
    sections = re.split(r"(\*\*[A-Z][A-Za-z ]+\*\*)", md_text)
    # sections will be like: ['', '**BACKGROUND**', 'content', '**ANOTHER SECTION**', 'content', ...]

    sections = filter_sections(sections, filename)

    # Pair headings with their content
    section_pairs = []
    for i in range(1, len(sections), 2):
        heading = sections[i].strip("*").strip()
        content = sections[i + 1].strip()
        if not heading:
            continue
        section_pairs.append((heading, content))

    # Write each section to a separate file
    for heading, content in section_pairs:
        # Create a safe filename from the heading
        safe_heading = heading.replace(" ", "_").lower()
        out_filename = os.path.join(out_dir, f"{safe_heading}.md")
        with open(out_filename, "w", encoding="utf-8") as f:
            f.write(f"**{heading}**\n\n{content}\n")


source_dir = os.path.expanduser("pdfs/")
for pdf_file in Path(source_dir).glob("*.pdf"):
    pdf_to_markdown(pdf_file)
