import nbformat
import os
import base64
from pathlib import Path
import sys

def extract_notebook_to_markdown_with_images(notebook_path, output_md_path=None):
    """
        Converts a Jupyter notebook to a Markdown file and exports images to a folder. 
        - Usage: python extract_notebook_to_markdown_with_images.py <notebook path.ipynb> <output path.md>
    """
    with open(notebook_path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)

    base_name = os.path.splitext(os.path.basename(notebook_path))[0]
    output_dir = os.path.dirname(output_md_path) if output_md_path else os.getcwd()
    image_dir = os.path.join(output_dir, f"{base_name}_images")
    Path(image_dir).mkdir(parents=True, exist_ok=True)

    output_lines = []
    image_count = 1

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
                        img_bytes = base64.b64decode(img_data)
                        img_filename = f"output_{image_count}.png"
                        img_path = os.path.join(image_dir, img_filename)
                        with open(img_path, 'wb') as img_file:
                            img_file.write(img_bytes)

                        # Insert image in Markdown
                        relative_path = os.path.relpath(img_path, output_dir).replace('\\', '/')
                        output_lines.append(f"![output_{image_count}]({relative_path})")
                        image_count += 1

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

    print(f"Markdown exported to {output_md_path}")
    print(f"Images saved to {image_dir}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python export_with_images.py notebook.ipynb [output.md]")
    else:
        notebook_path = sys.argv[1]
        output_md_path = sys.argv[2] if len(sys.argv) > 2 else None
        extract_notebook_to_markdown_with_images(notebook_path, output_md_path)
