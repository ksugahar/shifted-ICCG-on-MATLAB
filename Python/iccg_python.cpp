#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include "iccg.h"
#include <sstream>
#include <iostream>

namespace py = pybind11;

// Helper function to convert scipy CSR matrix data to our SparseMatrix format
SparseMatrix create_sparse_matrix_from_csr(
	py::array_t<double> data,
	py::array_t<int> indices,
	py::array_t<int> indptr,
	size_t shape0,
	size_t shape1) {
	
	auto data_ptr = static_cast<double*>(data.request().ptr);
	auto indices_ptr = static_cast<int*>(indices.request().ptr);
	auto indptr_ptr = static_cast<int*>(indptr.request().ptr);
	
	size_t nnz = data.size();
	size_t n_rows = shape0;
	
	SparseMatrix A;
	A.M = n_rows;
	A.N = shape1;
	A.nnz = nnz;
	
	// Copy data
	A.val.assign(data_ptr, data_ptr + nnz);
	
	// Convert indices to size_t
	A.col_idx.resize(nnz);
	for (size_t i = 0; i < nnz; ++i) {
		A.col_idx[i] = static_cast<size_t>(indices_ptr[i]);
	}
	
	// Convert indptr to size_t
	A.row_ptr.resize(n_rows + 1);
	for (size_t i = 0; i <= n_rows; ++i) {
		A.row_ptr[i] = static_cast<size_t>(indptr_ptr[i]);
	}
	
	return A;
}

// Result structure for Python
struct ICCGResult {
	py::array_t<double> x;		   // Solution vector
	int flag;						 // Convergence flag
	double relres;					// Relative residual
	int iterations;				   // Number of iterations
	int iter_best;					// Iteration where best solution was found
	py::array_t<double> residual_log; // Residual history
	double shift_used;				// Final shift parameter used
};

// Python wrapper for ICCG solver
ICCGResult solve_iccg(
	py::array_t<double> data,
	py::array_t<int> indices,
	py::array_t<int> indptr,
	py::array_t<double> b,
	double tol = 1e-6,
	int max_iter = 1000,
	double shift = 1.0,
	py::object x0 = py::none(),
	py::dict options = py::dict(),
	bool verbose = false) {
	
	// Get matrix dimensions from b vector
	auto b_info = b.request();
	if (b_info.ndim != 1) {
		throw std::runtime_error("b must be a 1D array");
	}
	size_t n = b_info.shape[0];
	
	// Create sparse matrix from CSR data
	SparseMatrix A = create_sparse_matrix_from_csr(data, indices, indptr, n, n);
	
	// Extract b vector
	auto b_ptr = static_cast<double*>(b_info.ptr);
	
	// Create options structure
	Options opts;
	
	// Parse options dictionary
	if (options.contains("diverge_factor")) {
		opts.diverge_factor = options["diverge_factor"].cast<double>();
	}
	if (options.contains("diverge_count")) {
		opts.diverge_count = options["diverge_count"].cast<int>();
	}
	if (options.contains("scaling")) {
		opts.use_scaling = options["scaling"].cast<bool>();
	}
	if (options.contains("max_shift_trials")) {
		opts.max_shift_trials = options["max_shift_trials"].cast<int>();
	}
	if (options.contains("shift_increment")) {
		opts.shift_increment = options["shift_increment"].cast<double>();
	}
	if (options.contains("max_shift_value")) {
		opts.max_shift_value = options["max_shift_value"].cast<double>();
	}
	if (options.contains("min_diagonal_threshold")) {
		opts.min_diagonal_threshold = options["min_diagonal_threshold"].cast<double>();
	}
	if (options.contains("zero_diagonal_replacement")) {
		opts.zero_diagonal_replacement = options["zero_diagonal_replacement"].cast<double>();
	}
	
	// Create solver
	ICCGSolver solver;
	solver.setCurrentShift(shift);
	
	// Prepare solution vector
	py::array_t<double> x_array({n});
	auto x_ptr = static_cast<double*>(x_array.request().ptr);
	
	// Set initial guess
	if (!x0.is_none()) {
		py::array_t<double> x0_array = x0.cast<py::array_t<double>>();
		auto x0_info = x0_array.request();
		if (x0_info.shape[0] != static_cast<py::ssize_t>(n)) {
			throw std::runtime_error("Initial guess x0 must have the same size as b");
		}
		auto x0_ptr = static_cast<double*>(x0_info.ptr);
		std::copy(x0_ptr, x0_ptr + n, x_ptr);
	} else {
		std::fill(x_ptr, x_ptr + n, 0.0);
	}
	
	// Prepare residual log
	std::vector<double> residual_log_vec(max_iter + 1);
	
	// Print initial info if verbose
	if (verbose) {
		std::cout << "ICCG Solver Parameters:" << std::endl;
		std::cout << "  Matrix size: " << n << " x " << n << std::endl;
		std::cout << "  Non-zeros: " << A.nnz << std::endl;
		std::cout << "  Tolerance: " << tol << std::endl;
		std::cout << "  Max iterations: " << max_iter << std::endl;
		std::cout << "  Initial shift: " << shift << std::endl;
		std::cout << "  Scaling: " << (opts.use_scaling ? "enabled" : "disabled") << std::endl;
		std::cout << "  Divergence factor: " << opts.diverge_factor << std::endl;
		std::cout << "  Divergence count: " << opts.diverge_count << std::endl;
	}
	
	// Call the solver
	double residual_norm;
	int iterations;
	int flag;
	int iter_best;
	
	incomplete_cholesky_conjugate_gradient(
		A, b_ptr, x_ptr, tol, max_iter, solver, opts,
		&residual_norm, &iterations, residual_log_vec.data(), &flag, &iter_best
	);
	
	// Print results if verbose
	if (verbose) {
		std::cout << "\nICCG Solver Results:" << std::endl;
		std::cout << "  Final shift: " << solver.getCurrentShift() << std::endl;
		std::cout << "  Iterations: " << iterations << std::endl;
		std::cout << "  Best iteration: " << iter_best << std::endl;
		std::cout << "  Relative residual: " << residual_norm << std::endl;
		std::cout << "  Flag: " << flag;
		switch(flag) {
			case 0: std::cout << " (Converged)" << std::endl; break;
			case 1: std::cout << " (Max iterations reached)" << std::endl; break;
			case 2: std::cout << " (Decomposition failed)" << std::endl; break;
			case 3: std::cout << " (Divergence detected)" << std::endl; break;
			default: std::cout << " (Unknown)" << std::endl;
		}
	}
	
	// Create residual log array (only up to actual iterations)
	py::array_t<double> residual_log_array({iterations + 1});
	auto residual_log_ptr = static_cast<double*>(residual_log_array.request().ptr);
	std::copy(residual_log_vec.begin(), residual_log_vec.begin() + iterations + 1, residual_log_ptr);
	
	// Return results
	ICCGResult result;
	result.x = x_array;
	result.flag = flag;
	result.relres = residual_norm;
	result.iterations = iterations;
	result.iter_best = iter_best;
	result.residual_log = residual_log_array;
	result.shift_used = solver.getCurrentShift();
	
	return result;
}

// Simplified interface that returns only the solution
py::array_t<double> iccg_solve(
	py::array_t<double> data,
	py::array_t<int> indices,
	py::array_t<int> indptr,
	py::array_t<double> b,
	double tol = 1e-6,
	int max_iter = 1000) {
	
	ICCGResult result = solve_iccg(data, indices, indptr, b, tol, max_iter);
	return result.x;
}

PYBIND11_MODULE(iccg_solver, m) {
	m.doc() = "Incomplete Cholesky Conjugate Gradient (ICCG) solver for sparse symmetric positive definite matrices";
	
	// Bind the Options class
	py::class_<Options>(m, "Options")
		.def(py::init<>())
		.def_readwrite("diverge_factor", &Options::diverge_factor)
		.def_readwrite("diverge_count", &Options::diverge_count)
		.def_readwrite("use_scaling", &Options::use_scaling)
		.def_readwrite("max_shift_trials", &Options::max_shift_trials)
		.def_readwrite("shift_increment", &Options::shift_increment)
		.def_readwrite("max_shift_value", &Options::max_shift_value)
		.def_readwrite("min_diagonal_threshold", &Options::min_diagonal_threshold)
		.def_readwrite("zero_diagonal_replacement", &Options::zero_diagonal_replacement);
	
	// Bind the ICCGResult class
	py::class_<ICCGResult>(m, "ICCGResult")
		.def_readonly("x", &ICCGResult::x, "Solution vector")
		.def_readonly("flag", &ICCGResult::flag, "Convergence flag (0=converged, 1=max_iter, 2=decomp_failed, 3=diverged)")
		.def_readonly("relres", &ICCGResult::relres, "Relative residual norm")
		.def_readonly("iterations", &ICCGResult::iterations, "Number of iterations performed")
		.def_readonly("iter_best", &ICCGResult::iter_best, "Iteration where best solution was found")
		.def_readonly("residual_log", &ICCGResult::residual_log, "History of relative residuals")
		.def_readonly("shift_used", &ICCGResult::shift_used, "Final shift parameter used");
	
	// Main solver function
	m.def("solve_iccg", &solve_iccg,
		py::arg("data"),
		py::arg("indices"),
		py::arg("indptr"),
		py::arg("b"),
		py::arg("tol") = 1e-6,
		py::arg("max_iter") = 1000,
		py::arg("shift") = 1.0,
		py::arg("x0") = py::none(),
		py::arg("options") = py::dict(),
		py::arg("verbose") = false,
		R"pbdoc(
		Solve a sparse symmetric positive definite linear system using ICCG.
		
		Parameters
		----------
		data : array_like
			CSR format data array (lower triangular part only)
		indices : array_like
			CSR format column indices array
		indptr : array_like
			CSR format row pointer array
		b : array_like
			Right-hand side vector
		tol : float, optional
			Convergence tolerance (default: 1e-6)
		max_iter : int, optional
			Maximum number of iterations (default: 1000)
		shift : float, optional
			Initial shift parameter (default: 1.0)
		x0 : array_like, optional
			Initial guess (default: zero vector)
		options : dict, optional
			Solver options dictionary with keys:
			- 'diverge_factor': float (default: 10.0)
			- 'diverge_count': int (default: 10)
			- 'scaling': bool (default: True)
			- 'max_shift_trials': int (default: 100)
			- 'shift_increment': float (default: 0.01)
			- 'max_shift_value': float (default: 5.0)
			- 'min_diagonal_threshold': float (default: 1e-6)
			- 'zero_diagonal_replacement': float (default: 1e-10)
		verbose : bool, optional
			Print solver information (default: False)
			
		Returns
		-------
		ICCGResult
			Object containing:
			- x: solution vector
			- flag: convergence flag
			- relres: relative residual
			- iterations: number of iterations
			- iter_best: best iteration
			- residual_log: residual history
			- shift_used: final shift parameter
		)pbdoc");
	
	// Simplified interface
	m.def("iccg_solve", &iccg_solve,
		py::arg("data"),
		py::arg("indices"),
		py::arg("indptr"),
		py::arg("b"),
		py::arg("tol") = 1e-6,
		py::arg("max_iter") = 1000,
		R"pbdoc(
		Simple interface to solve Ax=b using ICCG (returns only solution).
		
		Parameters
		----------
		data : array_like
			CSR format data array (lower triangular part only)
		indices : array_like
			CSR format column indices array
		indptr : array_like
			CSR format row pointer array
		b : array_like
			Right-hand side vector
		tol : float, optional
			Convergence tolerance (default: 1e-6)
		max_iter : int, optional
			Maximum number of iterations (default: 1000)
			
		Returns
		-------
		array_like
			Solution vector x
		)pbdoc");
}