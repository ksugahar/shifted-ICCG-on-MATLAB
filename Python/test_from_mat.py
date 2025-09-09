#!/usr/bin/env python3
"""
Test script that loads Ab.mat and performs ICCG solve similar to MATLAB test_iccg.m
"""

import numpy as np
import scipy.io as sio
import scipy.sparse as sp
import matplotlib.pyplot as plt
import time

def symmetric_sparse_to_csr(A):
	"""Convert symmetric sparse matrix to CSR format (lower triangular part only)."""
	# Extract lower triangular part (including diagonal)
	A_lower = sp.tril(A, format='csr')
	return A_lower.data, A_lower.indices, A_lower.indptr

def main():
	# Load Ab.mat file
	print("Loading Ab.mat...")
	mat_data = sio.loadmat('../Ab.mat')
	A = mat_data['A']
	b = mat_data['b'].flatten()
	
	# Convert to CSR format if needed
	if not sp.issparse(A):
		A = sp.csr_matrix(A)
	else:
		A = A.tocsr()
	
	N = A.shape[0]
	print(f"Matrix size: {N} x {N}")
	print(f"Non-zeros: {A.nnz}")
	
	# Convert symmetric sparse to CSR (lower triangular)
	vals, col_ind, row_ptr = symmetric_sparse_to_csr(A)
	
	# Setup plot
	plt.figure(figsize=(10, 8))
	plt.yscale('log')
	plt.xlabel('iteration')
	plt.ylabel('residual error')
	plt.grid(True, which='both', linestyle='-', alpha=0.3)
	plt.grid(True, which='minor', linestyle=':', alpha=0.2)
	
	labels = []
	
	# Test with different shift values
	for shift in [1.0, 2.0]:
		tol = 1e-9
		max_iter = 3000
		options = {
			'scaling': 1.0,
			'diverge_factor': 1.03,
			'diverge_count': 8
		}
		
		try:
			import iccg_solver
			
			# Initial guess (zeros)
			x0 = np.zeros(N)
			
			print(f"\nSolving with shift = {shift:.1f}...")
			start_time = time.time()
			
			# Solve using ICCG
			result = iccg_solver.solve_iccg(
				vals,
				col_ind,
				row_ptr,
				b,
				tol=tol,
				max_iter=max_iter,
				shift=shift,
				x0=x0,
				options=options,
				verbose=False
			)
			
			elapsed_time = time.time() - start_time
			
			# Extract results
			x = result.x
			flag = result.flag
			relres = result.relres
			iter_best = result.iter_best
			residual_log = result.residual_log
			shift_used = result.shift_used
			
			print(f"Python C++	elapsed time = {elapsed_time:.2f} sec")
			print(f"|x| = {np.linalg.norm(x):.2f},	|Ax-b|/|b| = {np.linalg.norm(A @ x - b)/np.linalg.norm(b):.3e},	iter = {iter_best},	shift = {shift_used:.2f}")
			
			# Plot convergence
			plt.semilogy(residual_log, '.-', markersize=8, label=f'Python C++ shift={shift_used:.2f}')
			labels.append(f'Python C++ shift={shift_used:.2f}')
			
		except ImportError as e:
			print(f"ERROR: iccg_solver module not found. Please build it first:")
			print("  python.exe -m pip install --upgrade pip setuptools wheel")
			print("  python.exe -m pip install pybind11 numpy scipy")
			print("  python.exe -m build")
			print("  or")
			print("  python.exe setup.py build_ext --inplace")
			return
		except Exception as e:
			print(f"ERROR during solve: {e}")
			return
	
	# Add legend and save plot
	plt.legend(labels, loc='upper right')
	plt.title('ICCG Convergence Comparison (Python)')
	plt.savefig('test_iccg_python.png', dpi=150, bbox_inches='tight')
	print("\nPlot saved as 'test_iccg_python.png'")
	plt.show()

if __name__ == "__main__":
	main()