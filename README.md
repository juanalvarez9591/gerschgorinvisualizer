# Gerschgorin Circles Visualizer


![image (1)](https://github.com/user-attachments/assets/75af44c9-3d53-4c90-b94d-c94b05ff8b46)


An interactive tool for visualizing! 
Gerschgorin's circle theorem and eigenvalue localization in matrices.

## Overview

The Gerschgorin Circles Visualizer is a Python application that allows users to explore the relationship between matrix entries, eigenvalues, and Gerschgorin circles. This tool provides intuitive visual representation of a fundamental theorem in linear algebra that helps locate eigenvalues of matrices.

## Features

- **Interactive Matrix Input**: Enter matrices of any reasonable dimension with support for both real and complex numbers
- **Example Library**: Choose from predefined example matrices to quickly explore different matrix types
- **Visualization**:
 - Color-coded Gerschgorin circles with centers, radii, and formulas
 - Eigenvalue plotting with clear markers
 - Automatic detection and highlighting of circle intersections
 - Equal-aspect plotting to maintain circle shapes
- **Information Panel**: Displays matrix values, eigenvalues, and circle specifications
- **Customization Options**: Toggle circle formulas on/off for cleaner visualization

## Requirements

- Python 3.x
- NumPy
- Matplotlib
- Tkinter (included with most Python installations)

## Installation

1. Ensure Python is installed on your system
2. Install the required packages:
3. Download the application source code

## Usage

1. Run the application:
2. Input your matrix:
- Set the matrix dimension and click "Generate Matrix Input"
- Enter the matrix values in the generated input fields
- Alternatively, select from predefined examples in the dropdown menu

3. Click "Plot Gerschgorin Circles" to generate the visualization

4. Interpret the results:
- Each row of the matrix corresponds to a colored circle
- Red stars mark the exact eigenvalue positions
- X markers indicate circle intersections
- The information panel provides detailed numerical data

## Input Format

The application accepts:
- Real numbers: `5`, `-3.2`
- Complex numbers: `3+2i`, `1-i`, `2.5+3.7i`
- Purely imaginary numbers: `i`, `-i`, `2.5i`

## Mathematical Background

Gerschgorin's circle theorem states that for an n×n matrix A, all eigenvalues lie within the union of n discs in the complex plane. Each disc D_i is centered at the diagonal element a_ii with radius equal to the sum of absolute values of other elements in that row:

D_i = {z ∈ ℂ : |z - a_ii| ≤ ∑_{j≠i} |a_ij|}

This application visualizes these discs and shows how they contain and localize the eigenvalues of the matrix.
