import nbformat

def print_cell(idx):
    path = "notebooks/wallet_selection/signal_lab/quickstart_signal_lab.ipynb"
    with open(path, "r") as f:
        nb = nbformat.read(f, as_version=4)
        
    print(f"--- Cell {idx} ---")
    print(nb.cells[idx].source)

if __name__ == "__main__":
    print_cell(12)
    print_cell(13)
    print_cell(14)
    print_cell(15)
