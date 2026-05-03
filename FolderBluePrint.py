import os
from pathlib import Path

def generate_tree_string(directory, prefix=""):
    """
    Recursively builds a string of the directory tree.
    """
    tree_str = ""
    path = Path(directory)
    
    # Folders to ignore to keep the markdown clean
    ignore_list = {'.git', '__pycache__', '.venv', '.pytest_cache', '.vscode'}
    
    # Get items, filter out ignored ones, and sort (folders first)
    items = [i for i in path.iterdir() if i.name not in ignore_list]
    items.sort(key=lambda x: (x.is_file(), x.name.lower()))
    
    for i, item in enumerate(items):
        is_last = (i == len(items) - 1)
        connector = "└── " if is_last else "├── "
        
        # Append current line to our string
        tree_str += f"{prefix}{connector}{item.name}\n"
        
        if item.is_dir():
            extension = "    " if is_last else "│   "
            tree_str += generate_tree_string(item, prefix + extension)
            
    return tree_str

def save_tree_to_desktop():
    # 1. Get user input for directory to scan
    target_input = input("Enter path to scan (or '.' for current): ") or "."
    target_path = Path(target_input).resolve()
    
    if not target_path.exists():
        print(f"❌ Error: {target_path} does not exist.")
        return

    # 2. Generate the content
    content = f"# Directory Tree: {target_path.name}\n\n"
    content += "```text\n"
    content += f"{target_path.name}/\n"
    content += generate_tree_string(target_path)
    content += "```\n"
    content += f"\n*Generated on: {Path(target_path).name}*"

    # 3. Define the Desktop path
    # Path.home() handles the /home/username part automatically
    desktop_path = Path.home() / "Desktop" / f"{target_path.name}_tree.md"

    # 4. Write the file
    try:
        desktop_path.write_text(content)
        print(f"✅ Success! Tree saved to: {desktop_path}")
    except Exception as e:
        print(f"❌ Failed to save file: {e}")

if __name__ == "__main__":
    save_tree_to_desktop()
