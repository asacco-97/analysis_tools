import nbformat
import os
import base64
import sys

def extract_notebook_to_markdown_inline_images(notebook_path, output_md_path=None):
    """
    Converts a Jupyter notebook to a Markdown file and embeds images as base64 data URIs.
    Usage: python extract_notebook_to_markdown_inline_images.py <notebook path.ipynb> [output path.md]
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    base_name = os.path.splitext(os.path.basename(notebook_path))[0]
    output_dir = os.path.dirname(output_md_path) if output_md_path else os.getcwd()

    output_lines = []

    for cell_index, cell in enumerate(nb.cells):
        if cell.cell_type == 'markdown':
            output_lines.append(cell.source)

        elif cell.cell_type == 'code':
            output_lines.append("```python")
            output_lines.append(cell.source)
            output_lines.append("```")

            for output in cell.get('outputs', []):
                if output.output_type == 'stream':
                    output_lines.append("**Output:**")
                    output_lines.append("```")
                    output_lines.append(output.get('text', '').strip())
                    output_lines.append("```")

                elif output.output_type == 'execute_result':
                    text = output.get('data', {}).get('text/plain', '').strip()
                    if text:
                        output_lines.append("**Result:**")
                        output_lines.append("```")
                        output_lines.append(text)
                        output_lines.append("```")

                elif output.output_type == 'display_data':
                    if 'image/png' in output.get('data', {}):
                        img_data = output['data']['image/png']
                        # Embed as base64 image
                        data_uri = f"data:image/png;base64,{img_data}"
                        output_lines.append(f"![image](data:image/png;base64,{img_data})")

                elif output.output_type == 'error':
                    output_lines.append("**Error:**")
                    output_lines.append("```")
                    output_lines.extend(output.get('traceback', []))
                    output_lines.append("```")

        output_lines.append("\n---\n")

    if not output_md_path:
        output_md_path = os.path.join(output_dir, f"{base_name}.md")

    with open(output_md_path, 'w', encoding='utf-8') as f:
        f.write("\n".join(output_lines))

    print(f"Markdown with inline images exported to {output_md_path}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_with_inline_images.py notebook.ipynb [output.md]")
    else:
        notebook_path = sys.argv[1]
        output_md_path = sys.argv[2] if len(sys.argv) > 2 else None
        extract_notebook_to_markdown_inline_images(notebook_path, output_md_path)