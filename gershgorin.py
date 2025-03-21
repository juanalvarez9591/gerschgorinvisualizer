import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.patches import Circle, FancyBboxPatch
import matplotlib.colors as mcolors
import tkinter as tk
from tkinter import ttk, messagebox
import re

class GerschgorinCirclesApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Gerschgorin Circles Visualizer")
        self.root.geometry("1200x800")
        
        # Create main frames
        self.input_frame = ttk.Frame(root, padding="10")
        self.input_frame.pack(side=tk.LEFT, fill=tk.Y)
        
        self.plot_frame = ttk.Frame(root)
        self.plot_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True)
        
        # Input section
        ttk.Label(self.input_frame, text="Matrix Dimension (n):", font=("Arial", 12)).grid(column=0, row=0, sticky=tk.W, pady=5)
        
        self.n_var = tk.StringVar(value="3")
        self.n_entry = ttk.Entry(self.input_frame, width=10, textvariable=self.n_var)
        self.n_entry.grid(column=1, row=0, sticky=tk.W, pady=5)
        
        self.generate_btn = ttk.Button(self.input_frame, text="Generate Matrix Input", command=self.generate_matrix_inputs)
        self.generate_btn.grid(column=0, row=1, columnspan=2, sticky=tk.W, pady=10)
        
        self.matrix_frame = ttk.Frame(self.input_frame)
        self.matrix_frame.grid(column=0, row=2, columnspan=2, sticky=tk.W)
        
        self.matrix_entries = []
        self.plot_button = None
        self.fig = None
        self.canvas = None
        
        # Add checkbox for showing formulas
        self.show_formulas_var = tk.BooleanVar(value=True)
        self.show_formulas_check = ttk.Checkbutton(
            self.input_frame, 
            text="Show Circle Formulas", 
            variable=self.show_formulas_var
        )
        self.show_formulas_check.grid(column=0, row=3, columnspan=2, sticky=tk.W, pady=5)
        
        # Examples dropdown
        ttk.Label(self.input_frame, text="Load Example:", font=("Arial", 12)).grid(column=0, row=4, sticky=tk.W, pady=20)
        
        self.examples = [
            "Custom Input",
            "2×2 Real Matrix",
            "3×3 Real Matrix",
            "3×3 Complex Matrix",
            "4×4 Diagonal Matrix"
        ]
        
        self.example_var = tk.StringVar()
        self.example_dropdown = ttk.Combobox(self.input_frame, textvariable=self.example_var, values=self.examples, width=20)
        self.example_dropdown.current(0)
        self.example_dropdown.grid(column=1, row=4, sticky=tk.W, pady=20)
        self.example_dropdown.bind("<<ComboboxSelected>>", self.load_example)
        
        # Initial setup
        self.generate_matrix_inputs()
        
        # Create matplotlib figure for the plot
        self.setup_plot()

    def setup_plot(self):
        # Create figure
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        self.fig.tight_layout()
        
        # Place the figure in the plot frame
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Initial message on plot
        self.ax.text(0.5, 0.5, "Enter matrix values and click 'Plot' to visualize", 
                    horizontalalignment='center', verticalalignment='center',
                    transform=self.ax.transAxes, fontsize=14)
        self.ax.set_xticks([])
        self.ax.set_yticks([])
        self.canvas.draw()

    def generate_matrix_inputs(self):
        try:
            n = int(self.n_var.get())
            if n <= 0:
                messagebox.showerror("Invalid Input", "Dimension must be a positive integer")
                return
            if n > 10:
                messagebox.showwarning("Large Matrix", "Large matrices may be difficult to display. Proceeding with n=" + str(n))
        except ValueError:
            messagebox.showerror("Invalid Input", "Please enter a valid integer for dimension")
            return
        
        # Clear previous matrix entries
        for widget in self.matrix_frame.winfo_children():
            widget.destroy()
        
        # Create new matrix entries
        self.matrix_entries = []
        
        # Header row for column indices
        ttk.Label(self.matrix_frame, text="", width=4).grid(column=0, row=0)
        for j in range(n):
            ttk.Label(self.matrix_frame, text=f"Col {j+1}", width=10).grid(column=j+1, row=0)
        
        for i in range(n):
            ttk.Label(self.matrix_frame, text=f"Row {i+1}").grid(column=0, row=i+1, sticky=tk.W)
            row_entries = []
            
            for j in range(n):
                entry = ttk.Entry(self.matrix_frame, width=12)
                entry.grid(column=j+1, row=i+1, padx=3, pady=3)
                entry.insert(0, "0")
                row_entries.append(entry)
                
            self.matrix_entries.append(row_entries)
        
        # Add plot button
        if self.plot_button:
            self.plot_button.destroy()
            
        self.plot_button = ttk.Button(self.input_frame, text="Plot Gerschgorin Circles", command=self.plot_circles)
        self.plot_button.grid(column=0, row=5, columnspan=2, pady=20)

    def parse_complex(self, s):
        """Parse a string into a complex number, handling various formats."""
        # Remove all spaces
        s = s.replace(" ", "")
        
        # Try direct complex conversion first
        try:
            # Replace 'i' with 'j' for Python's complex number notation
            return complex(s.replace('i', 'j'))
        except ValueError:
            pass
        
        # Try regex patterns
        # Pattern for "a+bi" or "a-bi" format
        pattern1 = r'([-+]?\d*\.?\d*)(?:([-+])(\d*\.?\d*)i)?'
        match = re.match(pattern1, s)
        
        if match:
            real_part = match.group(1)
            if real_part == '' or real_part == '+':
                real_part = '0'
            if real_part == '-':
                real_part = '-0'
            
            real = float(real_part)
            
            # Check if there's an imaginary part
            if match.group(2) and match.group(3):
                sign = match.group(2)
                imag_part = match.group(3)
                
                if imag_part == '':
                    imag_part = '1'
                    
                imag = float(imag_part)
                if sign == '-':
                    imag = -imag
                    
                return complex(real, imag)
            else:
                return complex(real, 0)
        
        # If all attempts fail, raise an exception
        raise ValueError(f"Could not parse '{s}' as a complex number")

    def get_matrix_from_entries(self):
        n = len(self.matrix_entries)
        matrix = np.zeros((n, n), dtype=complex)
        
        for i in range(n):
            for j in range(n):
                entry_text = self.matrix_entries[i][j].get().strip()
                try:
                    if entry_text:
                        matrix[i, j] = self.parse_complex(entry_text)
                    else:
                        matrix[i, j] = 0
                except ValueError as e:
                    messagebox.showerror("Invalid Input", f"Error at position ({i+1},{j+1}): {str(e)}")
                    return None
        
        return matrix

    def load_example(self, event=None):
        example = self.example_var.get()
        
        if example == "Custom Input":
            return
            
        elif example == "2×2 Real Matrix":
            # Set dimension to 2
            self.n_var.set("2")
            self.generate_matrix_inputs()
            
            # Example 2×2 matrix with real entries
            example_matrix = [
                [4, 1],
                [2, 3]
            ]
            
        elif example == "3×3 Real Matrix":
            # Set dimension to 3
            self.n_var.set("3")
            self.generate_matrix_inputs()
            
            # Example 3×3 matrix with real entries
            example_matrix = [
                [5, 2, 1],
                [1, 6, 3],
                [2, 1, 4]
            ]
            
        elif example == "3×3 Complex Matrix":
            # Set dimension to 3
            self.n_var.set("3")
            self.generate_matrix_inputs()
            
            # Example 3×3 matrix with complex entries
            example_matrix = [
                ["3+0i", "1+1i", "0+0i"],
                ["1-1i", "4+0i", "2+0i"],
                ["0+0i", "2+0i", "5+0i"]
            ]
            
        elif example == "4×4 Diagonal Matrix":
            # Set dimension to 4
            self.n_var.set("4")
            self.generate_matrix_inputs()
            
            # Example 4×4 diagonal matrix
            example_matrix = [
                [3, 0, 0, 0],
                [0, 5, 0, 0],
                [0, 0, 7, 0],
                [0, 0, 0, 9]
            ]
        
        # Fill the matrix entries with the example
        for i in range(len(example_matrix)):
            for j in range(len(example_matrix[0])):
                self.matrix_entries[i][j].delete(0, tk.END)
                self.matrix_entries[i][j].insert(0, str(example_matrix[i][j]))

    def format_complex(self, num):
        """Format a complex number for display in text."""
        if abs(num.imag) < 1e-10:  # Essentially a real number
            return f"{num.real:.2f}"
        elif abs(num.real) < 1e-10:  # Essentially an imaginary number
            if num.imag == 1:
                return "i"
            elif num.imag == -1:
                return "-i"
            else:
                return f"{num.imag:.2f}i"
        else:
            if num.imag > 0:
                return f"{num.real:.2f}+{num.imag:.2f}i"
            else:
                return f"{num.real:.2f}{num.imag:.2f}i"  # Minus sign is included in imag

    def gerschgorin_circles(self, matrix):
        """Calculate the Gerschgorin circles for a given matrix."""
        n = matrix.shape[0]
        centers = np.zeros(n, dtype=complex)
        radii = np.zeros(n)
        
        for i in range(n):
            centers[i] = matrix[i, i]
            radii[i] = np.sum(np.abs(matrix[i, :])) - np.abs(matrix[i, i])
        
        return centers, radii

    def plot_circles(self):
        matrix = self.get_matrix_from_entries()
        if matrix is None:
            return
        
        # Clear previous plot
        self.ax.clear()
        
        n = matrix.shape[0]
        centers, radii = self.gerschgorin_circles(matrix)
        eigenvalues = np.linalg.eigvals(matrix)
        
        # Get colormap for circles
        colors = list(mcolors.TABLEAU_COLORS.values())
        
        # Plot Gerschgorin circles
        circle_patches = []
        for i in range(n):
            color = colors[i % len(colors)]
            circle = Circle((centers[i].real, centers[i].imag), radii[i], 
                          alpha=0.2, color=color, 
                          label=f"Circle {i+1}")
            circle_patches.append(circle)
            self.ax.add_patch(circle)
            
            # Plot center point
            self.ax.plot(centers[i].real, centers[i].imag, 'o', color=color, markersize=6)
            
            # Add radius line
            angle = 45  # degrees
            dx = radii[i] * np.cos(np.radians(angle))
            dy = radii[i] * np.sin(np.radians(angle))
            self.ax.plot([centers[i].real, centers[i].real + dx], 
                       [centers[i].imag, centers[i].imag + dy], 
                       '-', color=color, linewidth=1.5)
            
            # Add radius label
            self.ax.text(centers[i].real + dx/2, centers[i].imag + dy/2, 
                      f"r = {radii[i]:.2f}", 
                      color=color, fontsize=9, 
                      ha='center', va='center',
                      bbox=dict(facecolor='white', alpha=0.7, boxstyle='round,pad=0.2'))
            
            # Add formula text if checkbox is selected
            if self.show_formulas_var.get():
                formula_text = f"|z-{self.format_complex(centers[i])}| ≤ {radii[i]:.2f}"
                # Position the formula text near the circle
                angle_text = 135  # degrees (different from radius angle)
                text_dist = radii[i] * 1.2  # Slightly outside the circle
                text_x = centers[i].real + text_dist * np.cos(np.radians(angle_text))
                text_y = centers[i].imag + text_dist * np.sin(np.radians(angle_text))
                
                # Add a background box to make text more readable
                self.ax.text(text_x, text_y, formula_text, 
                          color=color, fontsize=10, fontweight='bold',
                          ha='center', va='center',
                          bbox=dict(facecolor='white', alpha=0.9, boxstyle='round,pad=0.3'))
        
        # Find and highlight intersections
        for i in range(n):
            for j in range(i+1, n):
                dist = abs(centers[i] - centers[j])
                # Check if circles intersect
                if dist < radii[i] + radii[j] and dist > abs(radii[i] - radii[j]):
                    # Calculate intersection points
                    a = (radii[i]**2 - radii[j]**2 + dist**2) / (2 * dist)
                    h = np.sqrt(max(0, radii[i]**2 - a**2))  # Use max to avoid negative values due to floating point errors
                    
                    # Direction vector from center i to center j
                    direction = (centers[j] - centers[i]) / dist
                    
                    # Calculate intersection points
                    p1 = centers[i] + a * direction + 1j * h * direction
                    p2 = centers[i] + a * direction - 1j * h * direction
                    
                    # Plot intersection points
                    self.ax.plot(p1.real, p1.imag, 'X', color='black', markersize=8, 
                               label='Intersection' if i==0 and j==1 else "")
                    self.ax.plot(p2.real, p2.imag, 'X', color='black', markersize=8)
                    
                    # Label the intersections
                    self.ax.text(p1.real, p1.imag + 0.5, f"C{i+1}∩C{j+1}", fontsize=8, ha='center', va='bottom')
                    self.ax.text(p2.real, p2.imag - 0.5, f"C{i+1}∩C{j+1}", fontsize=8, ha='center', va='top')
        
        # Plot eigenvalues
        self.ax.plot(eigenvalues.real, eigenvalues.imag, 'r*', markersize=12, label='Eigenvalues')
        
        # Label eigenvalues
        for i, eig in enumerate(eigenvalues):
            self.ax.text(eig.real, eig.imag + 0.3, f"λ{i+1}", color='red', fontsize=10, ha='center', va='bottom')
        
        # Set axis labels and title
        self.ax.set_xlabel('Real', fontsize=12)
        self.ax.set_ylabel('Imaginary', fontsize=12)
        self.ax.set_title('Gerschgorin Circles, Intersections, and Eigenvalues', fontsize=14)
        
        # Add legend and grid
        handles, labels = self.ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        self.ax.legend(by_label.values(), by_label.keys(), loc='upper right')
        self.ax.grid(True)
        
        # Make axes equal to preserve circle shape
        self.ax.set_aspect('equal')
        
        # Set limits with some padding
        max_radius = max(radii) if len(radii) > 0 else 1
        max_abs_center = max([abs(c) for c in centers]) if len(centers) > 0 else 1
        max_abs_eigenval = max([abs(e) for e in eigenvalues]) if len(eigenvalues) > 0 else 1
        limit = max(max_abs_center + max_radius, max_abs_eigenval) * 1.2
        self.ax.set_xlim(-limit, limit)
        self.ax.set_ylim(-limit, limit)
        
        # Add a reference line for the real and imaginary axes
        self.ax.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        self.ax.axvline(x=0, color='k', linestyle='-', alpha=0.3)
        
        # Create a text box with matrix details
        matrix_text = "Matrix:\n"
        for i in range(n):
            row_text = "["
            for j in range(n):
                val = matrix[i, j]
                if j > 0:
                    row_text += ", "
                row_text += self.format_complex(val)
            row_text += "]"
            matrix_text += row_text + "\n"
        
        # Add eigenvalue information
        eigenvalue_text = "\nEigenvalues:\n"
        for i, eig in enumerate(eigenvalues):
            eigenvalue_text += f"λ{i+1} = {self.format_complex(eig)}\n"
        
        # Create a text box for Gerschgorin circles information
        circles_text = "\nGerschgorin Circles:\n"
        for i in range(n):
            circles_text += f"C{i+1}: |z-{self.format_complex(centers[i])}| ≤ {radii[i]:.4f}\n"
        
        # Combine all information
        info_text = matrix_text + eigenvalue_text + circles_text
        
        # Place the info text box in a good position
        self.ax.text(0.02, 0.98, info_text, transform=self.ax.transAxes,
                   verticalalignment='top', horizontalalignment='left',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.9),
                   fontsize=9)
        
        self.canvas.draw()

def main():
    root = tk.Tk()
    app = GerschgorinCirclesApp(root)
    root.mainloop()

if __name__ == "__main__":
    main()