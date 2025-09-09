#!/usr/bin/env python3
"""
Demonstration comparing MATLAB MEX function and Python binding usage.
This shows how to use the same underlying C++ implementation from both languages.
"""

import numpy as np
import scipy.sparse as sp

def create_demo_problem():
	"""Create a simple demo problem that can be used in both MATLAB and Python."""
	# Create a 5x5 symmetric positive definite matrix
	# This matches the example from the MATLAB documentation
	n = 5
	
	# Create matrix: A = [[4, -1, 0, 0, 0],
	#					 [-1, 4, -1, 0, 0], 
	#					 [0, -1, 4, -1, 0],
	#					 [0, 0, -1, 4, -1],
	#					 [0, 0, 0, -1, 4]]
	
	row_indices = [0, 1, 1, 2, 2, 3, 3, 4, 4]
	col_indices = [0, 0, 1, 1, 2, 2, 3, 3, 4]  
	data = [4, -1, 4, -1, 4, -1, 4, -1, 4]
	
	# Create lower triangular CSR matrix
	A_lower = sp.csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
	
	# Create full symmetric matrix for verification
	A_full = A_lower + A_lower.T - sp.diags(A_lower.diagonal())
	
	# Right-hand side vector
	b = np.array([1.0, 2.0, 3.0, 2.0, 1.0])
	
	print("Demo Problem:")
	print("=============")
	print("Matrix A (full symmetric):")
	print(A_full.toarray())
	print(f"\nRight-hand side b: {b}")
	print(f"Matrix size: {n}x{n}")
	print(f"Lower triangular non-zeros: {A_lower.nnz}")
	
	return A_lower, A_full, b

def demo_python_usage():
	"""Demonstrate Python usage of ICCG solver."""
	print("\n" + "="*60)
	print("Python Usage with iccg_solver")
	print("="*60)
	
	try:
		import iccg_solver
		
		A_lower, A_full, b = create_demo_problem()
		
		print("\nPython code:")
		print("------------")
		print("""
import iccg_solver

# Solve with detailed results
result = iccg_solver.solve_iccg(
	A_lower.data,	  # CSR data array
	A_lower.indices,   # CSR column indices  
	A_lower.indptr,	# CSR row pointers
	b,				 # Right-hand side
	tol=1e-8,		 # Convergence tolerance
	max_iter=100,	 # Maximum iterations
	verbose=True	  # Print solver info
)

x = result.x		  # Solution
flag = result.flag	# Convergence flag
""")
		
		print("Output:")
		print("-------")
		
		# Solve using ICCG
		result = iccg_solver.solve_iccg(
			A_lower.data,
			A_lower.indices,
			A_lower.indptr,
			b,
			tol=1e-8,
			max_iter=100,
			verbose=True
		)
		
		print(f"\nSolution x = {result.x}")
		print(f"Convergence flag = {result.flag} (0=converged)")
		print(f"Iterations = {result.iterations}")
		print(f"Final residual = {result.relres:.2e}")
		
		# Verify solution
		residual = np.linalg.norm(A_full @ result.x - b) / np.linalg.norm(b)
		print(f"Verification: ||Ax-b||/||b|| = {residual:.2e}")
		
		# Simple interface demo
		print("\nSimple interface:")
		print("-----------------")
		print("x = iccg_solver.iccg_solve(A_lower.data, A_lower.indices, A_lower.indptr, b)")
		x_simple = iccg_solver.iccg_solve(A_lower.data, A_lower.indices, A_lower.indptr, b)
		print(f"Solution x = {x_simple}")
		
		return result.x
		
	except ImportError:
		print("ERROR: iccg_solver not found. Build with:")
		print("  python setup.py build_ext --inplace")
		return None

def show_matlab_equivalent():
	"""Show equivalent MATLAB code."""
	print("\n" + "="*60)
	print("Equivalent MATLAB Usage")
	print("="*60)
	
	A_lower, A_full, b = create_demo_problem()
	
	print("\nMATLAB code:")
	print("------------")
	
	# Show how to create the data in MATLAB
	print("""
% Create the same problem in MATLAB
n = 5;
row_ind = [1, 2, 2, 3, 3, 4, 4, 5, 5];  % 1-based indexing
col_ind = [1, 1, 2, 2, 3, 3, 4, 4, 5];
vals = [4, -1, 4, -1, 4, -1, 4, -1, 4];

% Convert to CSR format (0-based indexing for MEX)
% or use 1-based indexing (auto-detected by MEX function)

b = [1; 2; 3; 2; 1];

% Solve using ICCG MEX function
options.scaling = true;
options.diverge_factor = 10.0;

[x, flag, relres, iter, residual_log, shift_used] = iccg_mex(...
	vals, col_ind, row_ind, b, 1e-8, 100, 1.0, [], options);

fprintf('Solution x = [%.6f, %.6f, %.6f, %.6f, %.6f]\\n', x);
fprintf('Flag = %d (0=converged)\\n', flag); 
fprintf('Iterations = %d\\n', iter);
fprintf('Residual = %.2e\\n', relres);
""")
	
	# Show the expected data arrays
	print(f"\nData arrays:")
	print(f"vals (lower triangular values) = {A_lower.data}")
	print(f"indices (column indices) = {A_lower.indices}")  
	print(f"indptr (row pointers) = {A_lower.indptr}")
	print(f"b (right-hand side) = {b}")

def compare_with_scipy():
	"""Compare ICCG with scipy solvers."""
	print("\n" + "="*60)
	print("Comparison with SciPy Solvers")
	print("="*60)
	
	A_lower, A_full, b = create_demo_problem()
	
	from scipy.sparse.linalg import spsolve, cg
	
	# Direct solver
	x_direct = spsolve(A_full, b)
	print(f"SciPy direct solve: x = {x_direct}")
	
	# Conjugate gradient
	x_cg, info = cg(A_full, b, tol=1e-8)
	print(f"SciPy CG: x = {x_cg}, info = {info}")
	
	# ICCG (if available)
	try:
		import iccg_solver
		result = iccg_solver.solve_iccg(
			A_lower.data, A_lower.indices, A_lower.indptr, b, 
			tol=1e-8, verbose=False)
		print(f"ICCG: x = {result.x}, flag = {result.flag}")
		
		# Compare solutions
		print(f"\nSolution differences:")
		print(f"||x_ICCG - x_direct|| = {np.linalg.norm(result.x - x_direct):.2e}")
		print(f"||x_ICCG - x_CG|| = {np.linalg.norm(result.x - x_cg):.2e}")
		
	except ImportError:
		print("ICCG solver not available")

def main():
	"""Run the demonstration."""
	print("ICCG Solver Demonstration")
	print("Comparing MATLAB and Python interfaces")
	
	# Create and show the demo problem
	create_demo_problem()
	
	# Show Python usage
	x_python = demo_python_usage()
	
	# Show MATLAB equivalent
	show_matlab_equivalent()
	
	# Compare with other solvers
	compare_with_scipy()
	
	print("\n" + "="*60)
	print("Summary")
	print("="*60)
	print("Both MATLAB and Python interfaces use the same C++ implementation:")
	print("- iccg.cpp: Core algorithm implementation")
	print("- iccg_mex.cpp: MATLAB MEX gateway")
	print("- iccg_python.cpp: Python pybind11 binding")
	print("\nThis ensures consistent results across both platforms!")

if __name__ == "__main__":
	main()