import nbformat

def update_notebook():
    path = "notebooks/wallet_selection/signal_lab/quickstart_signal_lab.ipynb"
    with open(path, "r") as f:
        nb = nbformat.read(f, as_version=4)
        
    # We want to insert the strategy imports and application early on.
    # Let's inspect the first few code cells.
    for i, cell in enumerate(nb.cells):
        if cell.cell_type == 'code':
            print(f"Cell {i}:\n{cell.source[:100]}...\n")

if __name__ == "__main__":
    update_notebook()
