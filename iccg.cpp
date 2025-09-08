#include "iccg.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <memory>
#include <limits>

// Vector operation functions (optimized version)
void vec_copy(const double* src, double* dst, size_t n) {
	std::memcpy(dst, src, n * sizeof(double));
}

double vec_dot(const double* x, const double* y, size_t n) {
	double sum = 0.0;
	for (size_t i = 0; i < n; i++) {
		sum += x[i] * y[i];
	}
	return sum;
}

void vec_axpy(double alpha, const double* x, double* y, size_t n) {
	// Compute y = alpha * x + y
	for (size_t i = 0; i < n; i++) {
		y[i] += alpha * x[i];
	}
}

void vec_scale(double alpha, double* x, size_t n) {
	for (size_t i = 0; i < n; i++) {
		x[i] *= alpha;
	}
}

// Sparse symmetric matrix-vector product: y = A * x (lower triangular part only stored)
// Calculate complete matrix-vector product using only the lower triangular part of symmetric matrix
void sparse_symm_matvec(const SparseMatrix& A, const double* x, double* y) {
	// Initialize output vector
	std::memset(y, 0, A.M * sizeof(double));
	// Symmetric matrix-vector product in compressed row storage format
	for (size_t i = 0; i < A.M; i++) {
		double sum = 0.0;
		size_t j_start = A.row_ptr[i];
		size_t j_end = A.row_ptr[i + 1];
		// For non-zero elements in row i
		for (size_t j = j_start; j < j_end; j++) {
			size_t col = A.col_idx[j];
			double aij = A.val[j];
			// Diagonal element
			if (col == i) {
				sum += aij * x[col];
			} else {
				// Off-diagonal element: process both matrix elements A[i,col] and A[col,i]
				sum += aij * x[col];  // Product of matrix element A[i,col] and vector element x[col]
				y[col] += aij * x[i];  // Product of matrix element A[col,i] and vector element x[i] (using symmetry)
			}
		}
		y[i] += sum;
	}
}

// Efficient matrix value retrieval helper with early termination
double get_matrix_value(const SparseMatrix& A, size_t row, size_t col) {
	// Early bounds check for efficiency
	if (row >= A.M || col > row) return 0.0; // Lower triangular matrix
	
	// Binary-like search within row (assuming sorted column indices)
	size_t start = A.row_ptr[row];
	size_t end = A.row_ptr[row + 1];
	
	for (size_t k = start; k < end; k++) {
		if (A.col_idx[k] == col) {
			return A.val[k];
		}
		// Early termination if we've passed the target column
		if (A.col_idx[k] > col) break;
	}
	return 0.0;
}

// Incomplete Cholesky decomposition that accurately reproduces the original algorithm (lower triangular input version)
bool ichol_original_algorithm(SparseMatrix& L, const SparseMatrix& A, ICCGSolver& solver, std::vector<double>& scaling, std::vector<double>& D_out, const Options& options) {
	size_t n = A.M;
	double shift = solver.getCurrentShift();
	
	// Adjust initial value of shift parameter
	if (shift < Constants::DEFAULT_SHIFT_VALUE) {
		shift = Constants::DEFAULT_SHIFT_VALUE;
		solver.setCurrentShift(shift);
	}
	
	// Input matrix already contains only lower triangular part, so calculate L matrix size directly
	size_t L_nnz = 0;
	for (size_t i = 0; i < n; i++) {
		bool has_diag = false;
		for (size_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
			size_t j = A.col_idx[k];
			// Input already contains only lower triangular part (column index j ≤ row index i)
			L_nnz++;
			if (j == i) has_diag = true;
		}
		if (!has_diag) L_nnz++;
	}
	
	// Allocate memory for lower triangular matrix L
	L.val.resize(L_nnz);
	L.col_idx.resize(L_nnz);
	L.row_ptr.resize(n + 1);
	L.M = n;
	L.N = n;
	L.nnz = L_nnz;
	
	// Copy structure and data from input matrix A to lower triangular matrix L
	L.row_ptr[0] = 0;
	size_t pos = 0;
	std::vector<double> original_diag(n);  // Save original diagonal elements for scaling
	
	for (size_t i = 0; i < n; i++) {
		bool has_diag = false;
		
		for (size_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
			size_t j = A.col_idx[k];
			// Input already contains only lower triangular part
			L.col_idx[pos] = j;
			L.val[pos] = A.val[k];
			if (j == i) {
				has_diag = true;
				original_diag[i] = A.val[k];  // Save original diagonal element
			}
			pos++;
		}
		
		if (!has_diag) {
			L.col_idx[pos] = i;
			L.val[pos] = 0.0;
			original_diag[i] = 0.0;
			pos++;
		}
		
		L.row_ptr[i + 1] = pos;
	}
	
	// Efficient scaling calculation using original diagonal values
	if (options.use_scaling) {
		scaling.resize(n);
		for (size_t i = 0; i < n; i++) {
			scaling[i] = (original_diag[i] > 0.0) ? 1.0 / sqrt(original_diag[i]) : 1.0;
		}
		// Note: Scaling will be applied dynamically during matrix restoration
	} else {
		// If not using scaling, use unit scaling
		scaling.resize(n, 1.0);
	}
	
	// Diagonal element array D (original algorithm doesn't explicitly initialize, but zero initialization is required)
	std::vector<double> D(n, 0.0);
	
	// Efficient matrix restoration function using helper
	auto restore_matrix = [&]() {
		for (size_t i = 0; i < n; i++) {
			for (size_t k = L.row_ptr[i]; k < L.row_ptr[i + 1]; k++) {
				size_t j = L.col_idx[k];
				double original_value = get_matrix_value(A, i, j);
				
				// Apply scaling if enabled
				if (options.use_scaling) {
					L.val[k] = original_value * scaling[i] * scaling[j];
				} else {
					L.val[k] = original_value;
				}
			}
		}
	};

	// Recalculation processing loop (as per original algorithm)
	bool restart;
	do {
		restart = false;
		
		// Efficient matrix restoration: compute values on-demand instead of copying arrays
		restore_matrix();
		
		size_t start_col_index = L.row_ptr[0];
		
		for (size_t i = 0; i < n; i++) {
			size_t end_col_index = L.row_ptr[i + 1] - 1;
			
			// Process off-diagonal elements
			for (size_t k = start_col_index; k < end_col_index; k++) {
				size_t current_col = L.col_idx[k];
				size_t col_start = L.row_ptr[current_col];
				size_t col_end = L.row_ptr[current_col + 1] - 1;
				
				// Inner loop of Original algorithm (accurately reproduces boundary conditions)
				for (size_t row_idx = start_col_index, col_idx = col_start; col_idx < col_end; ) {
					if (row_idx >= k) break;  // End when index reaches boundary
					
					size_t row_col = L.col_idx[row_idx];
					size_t col_col = L.col_idx[col_idx];
					
					if (col_col < row_col) {
						col_idx++;
					} else if (col_col > row_col) {
						row_idx++;
					} else {
						// Only when col_col < current_col (use only already computed D elements)
						if (col_col < current_col) {
							L.val[k] -= L.val[row_idx] * L.val[col_idx] / D[col_col];
						}
						row_idx++;
						col_idx++;
					}
				}
				// Use D[current_col] only when current_col < i (when already computed)
				if (current_col < i) {
					L.val[k] *= D[current_col];
				}
			}
			
			// Process diagonal elements (as per Original implementation)
			// Get original diagonal value and apply scaling on-demand
			double original_diag_value = original_diag[i];
			double diagonal_value = options.use_scaling ? original_diag_value * scaling[i] * scaling[i] : original_diag_value;
			double original_diagonal_value = diagonal_value;
			if (diagonal_value > 0.0) diagonal_value *= shift;
			
			for (size_t k = start_col_index; k < end_col_index; k++) {
				size_t col_index = L.col_idx[k];
				// Use only already computed D elements
				if (col_index < i) {
					diagonal_value -= L.val[k] * L.val[k] / D[col_index];
				}
			}
			
			// Shift adjustment judgment
			if (!solver.isShiftFixed()) {
				if (diagonal_value < options.min_diagonal_threshold && original_diagonal_value > 0.0 && shift < options.max_shift_value) {
					restart = true;
					shift += options.shift_increment;
					solver.setCurrentShift(shift);
					break;
				}
			}
			
			// Set D[i]
			if (diagonal_value <= 0.0) diagonal_value = options.zero_diagonal_replacement;
			diagonal_value = 1.0 / diagonal_value;
			D[i] = diagonal_value;
			
			start_col_index = end_col_index + 1;
		}
	} while (restart);
	
	// Store D array in output parameter
	D_out = D;
	
	return true;
}

// Apply preconditioning based on the Original algorithm
void apply_preconditioner_original(const SparseMatrix& L, const std::vector<double>& D, const std::vector<double>& scaling, const double* r, double* z, double* temp, const Options& options) {
	size_t n = L.M;
	
	static int call_count = 0;
	call_count++;
	
	// Scale input vector when using scaling
	if (options.use_scaling) {
		for (size_t i = 0; i < n; i++) {
			temp[i] = r[i] * scaling[i];
		}
	} else {
		// Copy directly when not using scaling
		for (size_t i = 0; i < n; i++) {
			temp[i] = r[i];
		}
	}
	
	// Forward substitution: solve L * y = temp (as per Original algorithm)
	size_t start_col_index = L.row_ptr[0];
	for (size_t i = 0; i < n; i++) {
		double accumulated_value = temp[i];
		size_t end_col_index = L.row_ptr[i + 1] - 1;
		
		for (size_t k = start_col_index; k < end_col_index; k++) {
			accumulated_value -= L.val[k] * z[L.col_idx[k]];
		}
		z[i] = accumulated_value;
		start_col_index = end_col_index + 1;
	}
	
	// Diagonal scaling using D array
	for (size_t i = 0; i < n; i++) {
		z[i] *= D[i];
	}
	
	// Backward substitution: solve L^T * y = z
	for (int i = static_cast<int>(n) - 1; i >= 0; i--) {
		size_t start_col_index = L.row_ptr[i];
		double current_value = z[i];
		
		for (size_t k = start_col_index; k < L.row_ptr[i + 1] - 1; k++) {
			z[L.col_idx[k]] -= L.val[k] * current_value;
		}
	}
	
	// Scale output vector when using scaling
	if (options.use_scaling) {
		for (size_t i = 0; i < n; i++) {
			z[i] *= scaling[i];
		}
	}
}

// Implementation of ICCG method (Original algorithm base + with scaling functionality)
void incomplete_cholesky_conjugate_gradient(const SparseMatrix& A, const double* b, double* x, 
											double tol, int max_iter, ICCGSolver& solver, const Options& options,
											double* residual_norm, int* iterations, double* residual_log, int* flag, int* iter_best) {

	size_t n = A.M;

	// Allocate working vectors
	std::vector<double> residual(n);	// Residual vector
	std::vector<double> preconditioned_residual(n);	// Preconditioned residual
	std::vector<double> search_direction(n);	// Search direction
	std::vector<double> matrix_times_search_direction(n);	// A * p
	std::vector<double> temp_vector(n);	// Temporary work
	std::vector<double> best_solution(n);	// Store best solution

	// Maximum number of shift adjustment trials (configurable)
	int max_shift_trials = options.max_shift_trials;
	bool decomposition_success = false;
	
	// L matrix and scaling
	SparseMatrix L_csr;
	std::vector<double> scaling;
	std::vector<double> D;
	
	// Shift adjustment loop
	for (int trial = 0; trial < max_shift_trials; trial++) {
		// Clear previous memory (for retry cases)
		if (trial > 0) {
			L_csr.val.clear();
			L_csr.col_idx.clear();
			L_csr.row_ptr.clear();
		}
		
		// Execute IC(0) incomplete Cholesky decomposition (Original algorithm)
		bool success = ichol_original_algorithm(L_csr, A, solver, scaling, D, options);
		
		if (success) {
			decomposition_success = true;
			break;
		} else {
			// Adjust shift and retry
			if (!solver.isShiftFixed() && solver.getCurrentShift() < options.max_shift_value) {
				double new_shift = solver.getCurrentShift() + options.shift_increment;
				solver.setCurrentShift(new_shift);
			} else {
				break;  // Adjustment not possible
			}
		}
	}
	
	if (!decomposition_success) {
		*flag = 2;  // Decomposition failed
		return;
	}

	// Diagonal element array D and scaling array are managed separately

	// Calculate initial residual: residual = b - A * x (symmetric matrix version)
	sparse_symm_matvec(A, x, residual.data());
	for (size_t i = 0; i < n; i++) {
		residual[i] = b[i] - residual[i];
	}

	// Initial preconditioned residual: preconditioned_residual = M^(-1) * residual
	apply_preconditioner_original(L_csr, D, scaling, residual.data(), preconditioned_residual.data(), temp_vector.data(), options);

	// Initial search direction: search_direction = preconditioned_residual
	vec_copy(preconditioned_residual.data(), search_direction.data(), n);

	// Initial inner products
	double residual_dot_preconditioned_old = vec_dot(residual.data(), preconditioned_residual.data(), n);
	double residual_dot_preconditioned_new;
	double initial_residual_norm = std::sqrt(vec_dot(residual.data(), residual.data(), n));
	double current_residual_norm = initial_residual_norm;

	// Divergence detection variables following original algorithm
	double initial_error = current_residual_norm / initial_residual_norm;  // Initial error (following original algorithm)
	double error_minimum = initial_error;  // Initialize with initial error
	vec_copy(x, best_solution.data(), n);
	int iteration_of_minimum_error = 0;  // Iteration count where best solution was found
	int divergence_counter = 0;  // Counter for divergence detection

	// Record residual history (adjusted so that iteration count=1 corresponds to array index=1)
	residual_log[0] = current_residual_norm / initial_residual_norm;  // Initial residual at iteration count 0

	// Incomplete Cholesky Conjugate Gradient iterations (following original algorithm: 1-based counter)
	int iteration;
	for (iteration = 1; iteration <= max_iter; iteration++) {
		// Matrix-vector product matrix_times_search_direction = A * search_direction (symmetric matrix version)
		sparse_symm_matvec(A, search_direction.data(), matrix_times_search_direction.data());
		
		// Step size alpha = (residual' * preconditioned_residual) / (search_direction' * A * search_direction)
		double search_direction_dot_matrix_product = vec_dot(search_direction.data(), matrix_times_search_direction.data(), n);
		double step_size = residual_dot_preconditioned_old / search_direction_dot_matrix_product;
		
		// Update solution vector x = x + alpha * search_direction
		vec_axpy(step_size, search_direction.data(), x, n);
		
		// Update residual vector residual = residual - alpha * A * search_direction
		vec_axpy(-step_size, matrix_times_search_direction.data(), residual.data(), n);
		
		// Preconditioned residual preconditioned_residual = M^(-1) * residual
		apply_preconditioner_original(L_csr, D, scaling, residual.data(), preconditioned_residual.data(), temp_vector.data(), options);
		
		// New inner product
		residual_dot_preconditioned_new = vec_dot(residual.data(), preconditioned_residual.data(), n);
		
		// Search direction coefficient beta = (residual_new' * preconditioned_residual_new) / (residual_old' * preconditioned_residual_old)
		double beta_coefficient = residual_dot_preconditioned_new / residual_dot_preconditioned_old;
		
		// Update search direction search_direction = preconditioned_residual + beta * search_direction
		vec_scale(beta_coefficient, search_direction.data(), n);
		vec_axpy(1.0, preconditioned_residual.data(), search_direction.data(), n);
		
		// Move to next iteration
		residual_dot_preconditioned_old = residual_dot_preconditioned_new;
		current_residual_norm = std::sqrt(vec_dot(residual.data(), residual.data(), n));
		
		// Record residual history (iteration count=1 corresponds to array index=1)
		residual_log[iteration] = current_residual_norm / initial_residual_norm;

		// Divergence detection identical to convergence judgment function from original algorithm
		double current_relative_residual = current_residual_norm / initial_residual_norm;
		
		// Initialization processing for iteration count == 1 (following original algorithm)
		if (iteration == 1) {
			divergence_counter = 0;
			error_minimum = initial_error;
		}
		
		if (current_relative_residual < error_minimum) {
			error_minimum = current_relative_residual;
			vec_copy(x, best_solution.data(), n);  // Save best solution
			iteration_of_minimum_error = iteration;
			divergence_counter = 0;  // Reset counter
		} else if (current_relative_residual < error_minimum * options.diverge_factor) {
			divergence_counter = 0;  // Within tolerance range, so reset counter
		} else {
			++divergence_counter;  // Outside tolerance range, so increment counter
		}
		
		// Termination condition: (relative error < convergence threshold || divergence flag counter > divergence detection count)
		if (current_relative_residual < tol || divergence_counter > options.diverge_count) {
			// As in the original algorithm, always restore the best solution upon termination
			vec_copy(best_solution.data(), x, n);
			
			if (current_relative_residual < tol) {
				// Termination due to convergence
				*flag = 0;
			} else {
				// Termination due to divergence
				*flag = 3;
			}
			break;
		}
	}

	// Also restore best solution when maximum iteration count is reached (same as termination function in original algorithm)
	if (iteration > max_iter) {
		vec_copy(best_solution.data(), x, n);
		*flag = 1;  // Maximum iteration count reached
	}

	// In Original, only the matrix is scaled, not the solution vector
	// So scaling restoration is not needed

	// Return results
	// Use iteration after exiting the loop as is, since it uses iteration after the Final() function in Original
	*residual_norm = ((*flag == 3 || *flag == 1) ? error_minimum : current_residual_norm / initial_residual_norm);
	*iterations = iteration;  // Actual iteration count (following Original)
	*iter_best = iteration_of_minimum_error;  // Iteration count of best solution
	if (*flag != 3 && *flag != 1) {
		*flag = 0;  // Normal convergence
	}

	// Memory is automatically freed by std::vector destructors
}