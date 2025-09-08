#!/usr/bin/env python3
"""
Test script for the ICCG Python binding.
This script demonstrates how to use the ICCG solver with scipy sparse matrices.
"""

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import spsolve
import matplotlib.pyplot as plt

def create_test_matrix(n=100, density=0.1):
    """Create a test sparse symmetric positive definite matrix."""
    # Create a random sparse matrix
    A = sp.random(n, n, density=density, format='csr', random_state=42)
    
    # Make it symmetric
    A = A + A.T
    
    # Make it positive definite by adding to diagonal
    A.setdiag(A.diagonal() + n * 0.1)
    
    # Extract only lower triangular part (including diagonal)
    A_lower = sp.tril(A, format='csr')
    
    return A, A_lower

def test_basic_solve():
    """Test basic ICCG solve functionality."""
    print("=" * 60)
    print("Testing basic ICCG solve")
    print("=" * 60)
    
    # Create test problem
    n = 50
    A, A_lower = create_test_matrix(n, density=0.2)
    b = np.random.rand(n)
    
    print(f"Matrix size: {n} x {n}")
    print(f"Non-zeros in full matrix: {A.nnz}")
    print(f"Non-zeros in lower triangular: {A_lower.nnz}")
    
    try:
        import iccg_solver
        
        # Solve using ICCG
        print("\nSolving with ICCG...")
        result = iccg_solver.solve_iccg(
            A_lower.data,
            A_lower.indices,
            A_lower.indptr,
            b,
            tol=1e-8,
            max_iter=500,
            verbose=True
        )
        
        # Verify solution
        x_iccg = result.x
        residual = np.linalg.norm(A @ x_iccg - b) / np.linalg.norm(b)
        
        print(f"\nICCG Results:")
        print(f"  Convergence flag: {result.flag}")
        print(f"  Iterations: {result.iterations}")
        print(f"  Best iteration: {result.iter_best}")
        print(f"  Final relative residual: {result.relres:.2e}")
        print(f"  Verification residual: {residual:.2e}")
        print(f"  Final shift: {result.shift_used:.3f}")
        
        # Compare with scipy direct solver
        print("\nComparing with scipy direct solver...")
        x_scipy = spsolve(A, b)
        error = np.linalg.norm(x_iccg - x_scipy) / np.linalg.norm(x_scipy)
        print(f"  Relative error vs scipy: {error:.2e}")
        
        # Plot convergence
        plt.figure(figsize=(10, 6))
        plt.semilogy(result.residual_log, 'b.-', label='ICCG residual')
        plt.axhline(y=1e-8, color='r', linestyle='--', label='Tolerance')
        plt.xlabel('Iteration')
        plt.ylabel('Relative residual')
        plt.title('ICCG Convergence History')
        plt.legend()
        plt.grid(True)
        plt.savefig('iccg_convergence.png', dpi=150, bbox_inches='tight')
        plt.show()
        
        return True
        
    except ImportError:
        print("ERROR: iccg_solver module not found. Please build it first:")
        print("  pip install pybind11")
        print("  python setup.py build_ext --inplace")
        return False

def test_with_options():
    """Test ICCG solver with different options."""
    print("\n" + "=" * 60)
    print("Testing ICCG with different options")
    print("=" * 60)
    
    try:
        import iccg_solver
        
        # Create test problem
        n = 30
        A, A_lower = create_test_matrix(n, density=0.3)
        b = np.random.rand(n)
        
        # Test with different options
        options_list = [
            {"scaling": True, "diverge_factor": 10.0},
            {"scaling": False, "diverge_factor": 5.0},
            {"scaling": True, "shift_increment": 0.05, "max_shift_value": 2.0}
        ]
        
        for i, options in enumerate(options_list):
            print(f"\nTest {i+1}: Options = {options}")
            
            result = iccg_solver.solve_iccg(
                A_lower.data,
                A_lower.indices,
                A_lower.indptr,
                b,
                tol=1e-6,
                max_iter=200,
                options=options,
                verbose=False
            )
            
            residual = np.linalg.norm(A @ result.x - b) / np.linalg.norm(b)
            print(f"  Flag: {result.flag}, Iterations: {result.iterations}")
            print(f"  Relative residual: {result.relres:.2e}")
            print(f"  Verification residual: {residual:.2e}")
            print(f"  Final shift: {result.shift_used:.3f}")
        
        return True
        
    except ImportError:
        print("ERROR: iccg_solver module not found.")
        return False

def test_simple_interface():
    """Test the simple iccg_solve interface."""
    print("\n" + "=" * 60)
    print("Testing simple iccg_solve interface")
    print("=" * 60)
    
    try:
        import iccg_solver
        
        # Create test problem
        n = 25
        A, A_lower = create_test_matrix(n, density=0.4)
        b = np.random.rand(n)
        
        # Solve using simple interface
        x = iccg_solver.iccg_solve(
            A_lower.data,
            A_lower.indices,
            A_lower.indptr,
            b,
            tol=1e-6,
            max_iter=100
        )
        
        # Verify solution
        residual = np.linalg.norm(A @ x - b) / np.linalg.norm(b)
        print(f"Simple interface - Relative residual: {residual:.2e}")
        
        return True
        
    except ImportError:
        print("ERROR: iccg_solver module not found.")
        return False

def benchmark_comparison():
    """Compare ICCG with other scipy solvers."""
    print("\n" + "=" * 60)
    print("Benchmark comparison with scipy solvers")
    print("=" * 60)
    
    try:
        import iccg_solver
        from scipy.sparse.linalg import cg, minres
        import time
        
        # Create larger test problem
        n = 200
        A, A_lower = create_test_matrix(n, density=0.05)
        b = np.random.rand(n)
        
        print(f"Matrix size: {n} x {n}, Non-zeros: {A.nnz}")
        
        # Benchmark ICCG
        start_time = time.time()
        result_iccg = iccg_solver.solve_iccg(
            A_lower.data,
            A_lower.indices,
            A_lower.indptr,
            b,
            tol=1e-6,
            max_iter=1000,
            verbose=False
        )
        time_iccg = time.time() - start_time
        residual_iccg = np.linalg.norm(A @ result_iccg.x - b) / np.linalg.norm(b)
        
        # Benchmark scipy CG
        start_time = time.time()
        x_cg, info_cg = cg(A, b, tol=1e-6, maxiter=1000)
        time_cg = time.time() - start_time
        residual_cg = np.linalg.norm(A @ x_cg - b) / np.linalg.norm(b)
        
        # Benchmark scipy MINRES
        start_time = time.time()
        x_minres, info_minres = minres(A, b, tol=1e-6, maxiter=1000)
        time_minres = time.time() - start_time
        residual_minres = np.linalg.norm(A @ x_minres - b) / np.linalg.norm(b)
        
        print(f"\nBenchmark Results:")
        print(f"{'Method':<12} {'Time (s)':<10} {'Iterations':<12} {'Residual':<12} {'Status':<10}")
        print("-" * 60)
        print(f"{'ICCG':<12} {time_iccg:<10.4f} {result_iccg.iterations:<12} {residual_iccg:<12.2e} {'OK' if result_iccg.flag == 0 else 'FAIL':<10}")
        print(f"{'scipy CG':<12} {time_cg:<10.4f} {1000 if info_cg != 0 else 'N/A':<12} {residual_cg:<12.2e} {'OK' if info_cg == 0 else 'FAIL':<10}")
        print(f"{'scipy MINRES':<12} {time_minres:<10.4f} {1000 if info_minres != 0 else 'N/A':<12} {residual_minres:<12.2e} {'OK' if info_minres == 0 else 'FAIL':<10}")
        
        return True
        
    except ImportError:
        print("ERROR: iccg_solver module not found.")
        return False

def main():
    """Run all tests."""
    print("ICCG Python Binding Test Suite")
    print("=" * 60)
    
    # List available tests
    tests = [
        ("Basic solve test", test_basic_solve),
        ("Options test", test_with_options),
        ("Simple interface test", test_simple_interface),
        ("Benchmark comparison", benchmark_comparison),
    ]
    
    results = []
    for test_name, test_func in tests:
        print(f"\nRunning: {test_name}")
        try:
            success = test_func()
            results.append((test_name, success))
        except Exception as e:
            print(f"ERROR in {test_name}: {e}")
            results.append((test_name, False))
    
    # Print summary
    print("\n" + "=" * 60)
    print("Test Summary")
    print("=" * 60)
    for test_name, success in results:
        status = "PASS" if success else "FAIL"
        print(f"{test_name:<30} {status}")
    
    total_pass = sum(1 for _, success in results if success)
    print(f"\nTotal: {total_pass}/{len(results)} tests passed")
    
    if total_pass == 0:
        print("\nTo build the Python extension:")
        print("  pip install pybind11 numpy scipy matplotlib")
        print("  python setup.py build_ext --inplace")

if __name__ == "__main__":
    main()