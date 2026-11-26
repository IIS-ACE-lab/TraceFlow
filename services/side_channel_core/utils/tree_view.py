# v3/UTS/scripts/tree_view.py
r"""
Print a colored, collapsible directory tree using Rich,
and optionally write it to a Markdown file.

Usage:
    python3 tree_view.py [root_path] [-o output.md]
    python tree_view.py -o Side\ Channel\ Analysis/doc/updated_tree.md
    python3 tree_view.py ../../../ -o ../../doc/SCA_tree.md
    python tree_view.py SCA -o services/doc/SCA_tree.md
    
    
If no root_path is provided, defaults to the parent directory of this script.

Author: Shraddha Mishra
"""


import os
import argparse

# Gracefully handle missing Rich dependency
try:
    from rich import print    # type: ignore
    from rich.tree import Tree  # type: ignore
    from rich.console import Console  # type: ignore
except ImportError:
    print("Error: The 'rich' library is not installed. Please install it with:\n    pip install rich")
    exit(1)

EXCLUDE_DIRS = {"__pycache__", ".github",  ".git", "logs", "generated_models", ".vscode"}  # Add any folder names to exclude

def build_tree(root_path: str) -> Tree:
    """Recursively build a Rich Tree for the given directory."""
    root_label = os.path.basename(root_path) or root_path
    tree = Tree(f"[bold blue]{root_label}[/]")
    _add_nodes(tree, root_path)
    return tree

def _add_nodes(branch: Tree, path: str):
    """Helper: add child nodes to a Rich Tree branch."""
    try:
        entries = sorted(os.listdir(path))
    except PermissionError:
        branch.add("[red]<permission denied>[/]")
        return

    for name in entries:
        full_path = os.path.join(path, name)
        
        # Skip excluded directories
        if name in EXCLUDE_DIRS and os.path.isdir(full_path):
            continue
        
        
        if os.path.isdir(full_path):
            child = branch.add(f"[bold cyan]{name}[/]")
            _add_nodes(child, full_path)
        else:
            branch.add(f"[green]{name}[/]")

def main():
    parser = argparse.ArgumentParser(description="Print a colored tree for a given directory (relative to where you run it).")
    parser.add_argument("folder", help="Folder to generate the tree for, relative to the current directory, e.g. '.' from the repository root")
    parser.add_argument("-o", "--output", help="Optional Markdown file to write the tree into")
    args = parser.parse_args()

    target_path = os.path.abspath(args.folder)

    if not os.path.isdir(target_path):
        print(f"[red]Error:[/] Folder '{args.folder}' does not exist at: {target_path}")
        exit(1)

    tree = build_tree(target_path)
    console = Console()
    console.print(tree)

    if args.output:
        # Render plain-text tree (no ANSI) into a markdown code block
        md_console = Console(record=True, force_terminal=False, width=1000)
        md_console.print(tree)
        text = md_console.export_text()

        # Ensure output goes inside the scanned folder
        output_path = os.path.join(target_path, os.path.basename(args.output))
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        with open(output_path, 'w') as out:
            out.write(f"The following is a tree view of the {args.folder} directory in the Markdown format:\n\n\n\n")
            for line in text.splitlines():
                out.write("\t" + line + "\n")

        print(f"[green]Tree written to {output_path}[/]")



if __name__ == "__main__":
    main()
