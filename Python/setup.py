from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

__version__ = "0.1.0"

ext_modules = [
    Pybind11Extension(
        "iccg_solver",
        ["iccg_python.cpp", "iccg.cpp"],
        include_dirs=["."],
        cxx_std=11,
        define_macros=[("VERSION_INFO", __version__)],
    ),
]

setup(
    name="iccg_solver",
    version=__version__,
    author="ICCG Development Team",
    description="Incomplete Cholesky Conjugate Gradient solver for sparse symmetric positive definite matrices",
    long_description="""
    ICCG Solver - Python Binding
    ============================
    
    This package provides a Python interface to the Incomplete Cholesky Conjugate Gradient (ICCG) 
    solver for sparse symmetric positive definite linear systems.
    
    Features:
    - Efficient C++ implementation
    - Compatible with scipy sparse matrices
    - Automatic shift adjustment for robustness
    - Optional diagonal scaling
    - Divergence detection
    
    Usage:
    ------
    ```python
    import numpy as np
    from scipy.sparse import csr_matrix
    import iccg_solver
    
    # Create a sparse symmetric positive definite matrix (lower triangular part)
    # and convert to CSR format
    A_lower = ...  # Your lower triangular matrix
    A_csr = csr_matrix(A_lower)
    
    # Right-hand side vector
    b = np.array([...])
    
    # Solve using ICCG
    result = iccg_solver.solve_iccg(
        A_csr.data, 
        A_csr.indices, 
        A_csr.indptr, 
        b,
        tol=1e-6,
        max_iter=1000
    )
    
    # Access solution
    x = result.x
    print(f"Converged: {result.flag == 0}")
    print(f"Iterations: {result.iterations}")
    print(f"Relative residual: {result.relres}")
    ```
    """,
    long_description_content_type="text/markdown",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.5.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
        ],
    },
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: C++",
        "Topic :: Scientific/Engineering :: Mathematics",
    ],
)