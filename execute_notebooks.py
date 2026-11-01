"""
execute_notebooks.py
Executes notebooks by running code cells in a shared namespace, capturing stdout,
and storing real executed outputs in the notebook JSON structure.
"""

import nbformat
import io
import sys
import os
import contextlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

def run_notebook(path):
    print(f"Executing {path}...")
    with open(path, 'r', encoding='utf-8') as f:
        nb = nbformat.read(f, as_version=4)
        
    global_ns = {'__name__': '__main__', '__file__': os.path.abspath(path)}
    
    # Custom display function to capture dataframe outputs
    def custom_display(*args):
        for a in args:
            print(a)
    global_ns['display'] = custom_display

    cell_exec_count = 1
    for cell in nb.cells:
        if cell.cell_type == 'code':
            source = cell.source
            stdout_capture = io.StringIO()
            stderr_capture = io.StringIO()
            
            with contextlib.redirect_stdout(stdout_capture), contextlib.redirect_stderr(stderr_capture):
                try:
                    exec(source, global_ns)
                except Exception as e:
                    print(f"Error in cell:\\n{source[:100]}...\\n{e}", file=stderr_capture)
                    raise
            
            output_text = stdout_capture.getvalue()
            error_text = stderr_capture.getvalue()
            combined_output = output_text + (("\\n[STDERR]\\n" + error_text) if error_text else "")
            
            cell.outputs = []
            if combined_output.strip():
                cell.outputs.append(nbformat.v4.new_output(
                    output_type='stream',
                    name='stdout',
                    text=combined_output
                ))
            cell.execution_count = cell_exec_count
            cell_exec_count += 1
            plt.close('all')

    with open(path, 'w', encoding='utf-8') as f:
        nbformat.write(nb, f)
    print(f"Successfully executed and saved {path}")

if __name__ == '__main__':
    notebooks = [
        "notebooks/01_Data_Cleaning_EDA.ipynb",
        "notebooks/02_Delivery_Delay_Prediction.ipynb"
    ]
    for nb in notebooks:
        run_notebook(nb)
