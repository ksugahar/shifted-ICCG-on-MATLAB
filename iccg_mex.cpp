#include "mex.h"
#include "matrix.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>
#include <memory>
#include <limits>
#include <stdexcept>

// Error handling utilities
namespace ErrorHandling {
	void validateInput(const char* param_name, const mxArray* array, bool should_be_real = true, bool should_be_scalar = false) {
		if (should_be_scalar && !mxIsScalar(array)) {
			mexErrMsgIdAndTxt("ICCG:invalidInput", "Parameter '%s' must be a scalar value", param_name);
		}
		if (should_be_real && (mxIsComplex(array) || !mxIsDouble(array))) {
			mexErrMsgIdAndTxt("ICCG:invalidInput", "Parameter '%s' must be a real double array", param_name);
		}
	}
	
	void validateDimensions(const char* param_name, const mxArray* array, mwSize expected_rows, mwSize expected_cols) {
		if (mxGetM(array) != expected_rows || mxGetN(array) != expected_cols) {
			mexErrMsgIdAndTxt("ICCG:dimensionMismatch", 
				"Parameter '%s' has incorrect dimensions: expected %lux%lu, got %lux%lu", 
				param_name, (unsigned long)expected_rows, (unsigned long)expected_cols, 
				(unsigned long)mxGetM(array), (unsigned long)mxGetN(array));
		}
	}
	
	void validateArraySize(const char* param_name, mwSize actual_size, mwSize expected_size) {
		if (actual_size != expected_size) {
			mexErrMsgIdAndTxt("ICCG:dimensionMismatch", 
				"Parameter '%s' has incorrect size: expected %lu elements, got %lu elements", 
				param_name, (unsigned long)expected_size, (unsigned long)actual_size);
		}
	}
	
	void throwError(const char* error_id, const char* message) {
		mexErrMsgIdAndTxt(error_id, "%s", message);
	}
}

// Safe type conversion utilities
namespace SafeTypeConversion {
	// Safe conversion from double to int with overflow/underflow checking
	int toInt(double value, const char* param_name) {
		// Check for NaN and infinity
		if (!std::isfinite(value)) {
			ErrorHandling::throwError("ICCG:invalidInput", 
				"Parameter contains invalid value (NaN or infinity)");
		}
		
		// Check for fractional part
		if (std::fabs(value - std::round(value)) > 1e-10) {
			ErrorHandling::throwError("ICCG:invalidInput", 
				"Parameter must be an integer value");
		}
		
		// Check for overflow/underflow
		if (value > std::numeric_limits<int>::max() || value < std::numeric_limits<int>::min()) {
			ErrorHandling::throwError("ICCG:overflow", 
				"Parameter value exceeds integer range");
		}
		
		return static_cast<int>(std::round(value));
	}
	
	// Safe conversion from double to mwIndex with bounds checking
	mwIndex toMwIndex(double value, const char* param_name) {
		// Check for NaN and infinity
		if (!std::isfinite(value)) {
			ErrorHandling::throwError("ICCG:invalidInput", 
				"Index contains invalid value (NaN or infinity)");
		}
		
		// Check for fractional part
		if (std::fabs(value - std::round(value)) > 1e-10) {
			ErrorHandling::throwError("ICCG:invalidInput", 
				"Index must be an integer value");
		}
		
		// Check for negative values (mwIndex is unsigned)
		if (value < 0.0) {
			ErrorHandling::throwError("ICCG:invalidInput", 
				"Index cannot be negative");
		}
		
		// Check for overflow
		if (value > std::numeric_limits<mwIndex>::max()) {
			ErrorHandling::throwError("ICCG:overflow", 
				"Index value exceeds maximum allowed value");
		}
		
		return static_cast<mwIndex>(std::round(value));
	}
	
	// Safe conversion from int to double (always safe, but explicit)
	double toDouble(int value) {
		return static_cast<double>(value);
	}
	
	// Validate and convert MATLAB scalar to positive integer
	int toPositiveInt(const mxArray* array, const char* param_name) {
		if (!mxIsScalar(array)) {
			ErrorHandling::throwError("ICCG:invalidInput", 
				"Parameter must be a scalar value");
		}
		
		double value = mxGetScalar(array);
		int result = toInt(value, param_name);
		
		if (result <= 0) {
			ErrorHandling::throwError("ICCG:invalidInput", 
				"Parameter must be positive");
		}
		
		return result;
	}
	
	// Safe array index conversion with bounds checking
	void convertIndicesArray(const double* source, std::vector<mwIndex>& target, 
							mwSize count, bool subtract_one, const char* array_name) {
		target.resize(count);
		for (mwSize i = 0; i < count; i++) {
			double value = subtract_one ? source[i] - 1.0 : source[i];
			
			// Validate index range for CSR format
			if (subtract_one && source[i] < 1.0) {
				ErrorHandling::throwError("ICCG:invalidInput", 
					"MATLAB indices must be >= 1 for 1-based indexing");
			}
			
			target[i] = toMwIndex(value, array_name);
		}
	}
	
	// Safe iteration bounds checking
	template<typename IndexType>
	void validateIterationCount(IndexType iteration, IndexType max_iterations, const char* context) {
		if (iteration < 0) {
			ErrorHandling::throwError("ICCG:invalidState", 
				"Iteration count cannot be negative");
		}
		if (iteration > max_iterations && max_iterations > 0) {
			ErrorHandling::throwError("ICCG:invalidState", 
				"Iteration count exceeds maximum allowed iterations");
		}
	}
}

// Algorithm constants
namespace Constants {
	// Incomplete Cholesky decomposition constants
	const double DEFAULT_SHIFT_VALUE = 1.0;           // Default shift parameter
	const double SHIFT_INCREMENT = 0.01;              // Shift increment for adjustment
	const double MAX_SHIFT_VALUE = 5.0;               // Maximum allowed shift value
	const double MIN_DIAGONAL_THRESHOLD = 1.e-6;      // Minimum diagonal threshold for shift adjustment
	const double ZERO_DIAGONAL_REPLACEMENT = 1e-10;   // Replacement value for zero/negative diagonal
	const double RECIPROCAL_THRESHOLD = 1e-10;        // Minimum value before taking reciprocal
	
	// Algorithm control constants
	const int DEFAULT_MAX_SHIFT_TRIALS = 100;         // Default maximum shift adjustment trials
	const int DEFAULT_DIVERGE_COUNT = 10;             // Default divergence detection count
	const double DEFAULT_DIVERGE_FACTOR = 10.0;       // Default divergence detection coefficient
	const bool DEFAULT_USE_SCALING = true;            // Default scaling setting
}

/*
 * ICCG_MEX - Incomplete Cholesky Conjugate Gradient Method for Sparse Linear Systems
 *
 * Solves the sparse symmetric positive definite linear system Ax = b using
 * the Incomplete Cholesky Conjugate Gradient (ICCG) method with optional scaling.
 *
 * SYNTAX:
 *   [x, flag, relres, iter, residual_log, shift_used] = iccg_mex(vals, col_ind, row_ptr, b, tol, max_iter, shift, x0, options)
 *
 * INPUTS:
 *   vals     - Double array of non-zero values (lower triangular part only)
 *   col_ind  - Double array of column indices (1-based or 0-based)
 *   row_ptr  - Double array of row pointers (1-based or 0-based)
 *   b        - Right-hand side vector (double column vector)
 *   tol      - Convergence tolerance (double scalar)
 *   max_iter - Maximum number of iterations (double scalar)
 *   shift    - Initial shift parameter for incomplete Cholesky decomposition (double scalar)
 *   x0       - Initial guess vector (double column vector, can be empty [])
 *   options  - Structure with optional parameters:
 *              .diverge_factor - Divergence detection coefficient (default: 10.0)
 *              .diverge_count  - Divergence detection count (default: 10)
 *              .scaling        - Enable/disable scaling (logical or numeric, default: true)
 *
 * OUTPUTS:
 *   x            - Solution vector (double column vector)
 *   flag         - Convergence flag:
 *                  0 = Normal convergence
 *                  1 = Maximum iterations reached
 *                  2 = Incomplete Cholesky decomposition failed
 *                  3 = Divergence detected
 *   relres       - Relative residual at termination
 *   iter         - Iteration count where best solution was found
 *   residual_log - Residual history for all iterations (including iteration 0)
 *   shift_used   - Final shift parameter used
 *
 * NOTES:
 *   - Input matrix A must be symmetric positive definite
 *   - Only the lower triangular part of A should be provided
 *   - Indices can be 1-based (MATLAB style) or 0-based (C style) - automatically detected
 *   - The algorithm implements the original ICCG method with shift adjustment and scaling
 *
 * EXAMPLE:
 *   % Create a sparse symmetric matrix (lower triangular part)
 *   A = sparse([1 2 2 3], [1 1 2 3], [4 -1 4 2], 3, 3);
 *   b = [1; 2; 3];
 *   [vals, col_ind, row_ptr] = find_lower_triangular(A);
 *   
 *   % Solve using ICCG
 *   options.scaling = true;
 *   [x, flag, relres, iter] = iccg_mex(vals, col_ind, row_ptr, b, 1e-6, 100, 1.0, [], options);
 *
 * See also: PCG, ICHOL, SPARSE
 */

// Options structure (original algorithm compliant with extended configurability)
struct Options {
	// Divergence detection parameters
	double diverge_factor;  // Divergence detection coefficient
	int diverge_count;      // Divergence detection count
	bool use_scaling;       // Control whether to use scaling
	
	// Algorithm control parameters  
	int max_shift_trials;           // Maximum shift adjustment trials
	double shift_increment;         // Shift increment for adjustment
	double max_shift_value;         // Maximum allowed shift value
	double min_diagonal_threshold;  // Minimum diagonal threshold for shift adjustment
	double zero_diagonal_replacement; // Replacement value for zero/negative diagonal
	
	Options() : 
		diverge_factor(Constants::DEFAULT_DIVERGE_FACTOR),
		diverge_count(Constants::DEFAULT_DIVERGE_COUNT),
		use_scaling(Constants::DEFAULT_USE_SCALING),
		max_shift_trials(Constants::DEFAULT_MAX_SHIFT_TRIALS),
		shift_increment(Constants::SHIFT_INCREMENT),
		max_shift_value(Constants::MAX_SHIFT_VALUE),
		min_diagonal_threshold(Constants::MIN_DIAGONAL_THRESHOLD),
		zero_diagonal_replacement(Constants::ZERO_DIAGONAL_REPLACEMENT)
	{}
};

// Compressed row storage format structure for sparse matrices
struct SparseMatrix {
	std::vector<double> val;	// Values of non-zero elements
	std::vector<mwIndex> row_ptr;	// Row pointers
	std::vector<mwIndex> col_idx;	// Column indices
	mwSize M;	// Number of rows
	mwSize N;	// Number of columns
	mwSize nnz;	// Number of non-zero elements
};

// Incomplete Cholesky Conjugate Gradient solver class
class ICCGSolver {
private:
	double shift;
	double dShift;
	double shiftMax;
	bool ifix;

public:
	ICCGSolver() : 
		shift(Constants::DEFAULT_SHIFT_VALUE), 
		dShift(Constants::SHIFT_INCREMENT), 
		shiftMax(Constants::MAX_SHIFT_VALUE), 
		ifix(false) {}
	
	void SetShiftParameter(double shift_val) {
		this->shift = shift_val;
		this->ifix = true;
	}
	
	double GetShiftParameter() const {
		return shift;
	}
	
	void FixShiftParameter(bool fix) {
		this->ifix = fix;
	}
	
	double getCurrentShift() const { return shift; }
	void setCurrentShift(double s) { shift = s; }
	bool isShiftFixed() const { return ifix; }
};

// Vector operation functions (optimized version)
inline void vec_copy(const double* src, double* dst, mwSize n) {
	std::memcpy(dst, src, n * sizeof(double));
}

inline double vec_dot(const double* x, const double* y, mwSize n) {
	double sum = 0.0;
	for (mwSize i = 0; i < n; i++) {
		sum += x[i] * y[i];
	}
	return sum;
}

inline void vec_axpy(double alpha, const double* x, double* y, mwSize n) {
	// Compute y = alpha * x + y
	for (mwSize i = 0; i < n; i++) {
		y[i] += alpha * x[i];
	}
}

inline void vec_scale(double alpha, double* x, mwSize n) {
	for (mwSize i = 0; i < n; i++) {
		x[i] *= alpha;
	}
}

// Sparse symmetric matrix-vector product: y = A * x (lower triangular part only stored)
// Calculate complete matrix-vector product using only the lower triangular part of symmetric matrix
inline void sparse_symm_matvec(const SparseMatrix& A, const double* x, double* y) {
	// Initialize output vector
	std::memset(y, 0, A.M * sizeof(double));
	// Symmetric matrix-vector product in compressed row storage format
	for (mwSize i = 0; i < A.M; i++) {
		double sum = 0.0;
		mwIndex j_start = A.row_ptr[i];
		mwIndex j_end = A.row_ptr[i + 1];
		// For non-zero elements in row i
		for (mwIndex j = j_start; j < j_end; j++) {
			mwIndex col = A.col_idx[j];
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
inline double get_matrix_value(const SparseMatrix& A, mwSize row, mwSize col) {
	// Early bounds check for efficiency
	if (row >= A.M || col > row) return 0.0; // Lower triangular matrix
	
	// Binary-like search within row (assuming sorted column indices)
	mwIndex start = A.row_ptr[row];
	mwIndex end = A.row_ptr[row + 1];
	
	for (mwIndex k = start; k < end; k++) {
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
	mwSize n = A.M;
	double shift = solver.getCurrentShift();
	
	// Adjust initial value of shift parameter
	if (shift < Constants::DEFAULT_SHIFT_VALUE) {
		shift = Constants::DEFAULT_SHIFT_VALUE;
		solver.setCurrentShift(shift);
	}
	
	// Input matrix already contains only lower triangular part, so calculate L matrix size directly
	mwSize L_nnz = 0;
	for (mwSize i = 0; i < n; i++) {
		bool has_diag = false;
		for (mwIndex k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
			mwIndex j = A.col_idx[k];
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
	mwIndex pos = 0;
	std::vector<double> original_diag(n);  // Save original diagonal elements for scaling
	
	for (mwSize i = 0; i < n; i++) {
		bool has_diag = false;
		
		for (mwIndex k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
			mwIndex j = A.col_idx[k];
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
		for (mwSize i = 0; i < n; i++) {
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
		for (mwSize i = 0; i < n; i++) {
			for (mwIndex k = L.row_ptr[i]; k < L.row_ptr[i + 1]; k++) {
				mwIndex j = L.col_idx[k];
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
		
		mwIndex start_col_index = L.row_ptr[0];
		
		for (mwSize i = 0; i < n; i++) {
			mwIndex end_col_index = L.row_ptr[i + 1] - 1;
			
			// Process off-diagonal elements
			for (mwIndex k = start_col_index; k < end_col_index; k++) {
				mwIndex current_col = L.col_idx[k];
				mwIndex col_start = L.row_ptr[current_col];
				mwIndex col_end = L.row_ptr[current_col + 1] - 1;
				
				// Inner loop of Original algorithm (accurately reproduces boundary conditions)
				for (mwIndex row_idx = start_col_index, col_idx = col_start; col_idx < col_end; ) {
					if (row_idx >= k) break;  // End when index reaches boundary
					
					mwIndex row_col = L.col_idx[row_idx];
					mwIndex col_col = L.col_idx[col_idx];
					
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
			
			for (mwIndex k = start_col_index; k < end_col_index; k++) {
				mwIndex col_index = L.col_idx[k];
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
	mwSize n = L.M;
	
	static int call_count = 0;
	call_count++;
	
	// Scale input vector when using scaling
	if (options.use_scaling) {
		for (mwSize i = 0; i < n; i++) {
			temp[i] = r[i] * scaling[i];
		}
	} else {
		// Copy directly when not using scaling
		for (mwSize i = 0; i < n; i++) {
			temp[i] = r[i];
		}
	}
	
	// Forward substitution: solve L * y = temp (as per Original algorithm)
	mwIndex start_col_index = L.row_ptr[0];
	for (mwSize i = 0; i < n; i++) {
		double accumulated_value = temp[i];
		mwIndex end_col_index = L.row_ptr[i + 1] - 1;
		
		for (mwIndex k = start_col_index; k < end_col_index; k++) {
			accumulated_value -= L.val[k] * z[L.col_idx[k]];
		}
		z[i] = accumulated_value;
		start_col_index = end_col_index + 1;
	}
	
	// Diagonal scaling using D array
	for (mwSize i = 0; i < n; i++) {
		z[i] *= D[i];
	}
	
	// Backward substitution: solve L^T * y = z
	mwIndex end_col_index = L.row_ptr[n] - 1;
	for (int i = n - 1; i >= 0; i--) {
		mwIndex start_col_index = L.row_ptr[i];
		double current_value = z[i];
		
		for (mwIndex k = start_col_index; k < L.row_ptr[i + 1] - 1; k++) {
			z[L.col_idx[k]] -= L.val[k] * current_value;
		}
		end_col_index = start_col_index - 1;
	}
	
	// Scale output vector when using scaling
	if (options.use_scaling) {
		for (mwSize i = 0; i < n; i++) {
			z[i] *= scaling[i];
		}
	}
}

// Implementation of ICCG method (Original algorithm base + with scaling functionality)
void incomplete_cholesky_conjugate_gradient(const SparseMatrix& A, const double* b, double* x, 
											double tol, int max_iter, ICCGSolver& solver, const Options& options,
											double* residual_norm, int* iterations, double* residual_log, int* flag, int* iter_best) {

	mwSize n = A.M;

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
	for (mwSize i = 0; i < n; i++) {
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
	double error_minimum;  // Set within convergence judgment function
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

// MEX gateway function
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

	// Validate number of input and output arguments
	if (nrhs != 9) {
		ErrorHandling::throwError("ICCG:invalidNumInputs", 
			"Expected exactly 9 input arguments.\n"
			"Usage: [x, flag, relres, iter, residual_log, shift_used] = iccg_mex(vals, col_ind, row_ptr, b, tol, max_iter, shift, x0, options)\n"
			"Note: vals and col_ind should contain only the lower triangular part of the symmetric matrix");
	}

	if (nlhs > 6) {
		ErrorHandling::throwError("ICCG:invalidNumOutputs", "Maximum 6 output arguments allowed");
	}

	// Validate and extract CSR format data for matrix A
	// vals: values of non-zero elements
	ErrorHandling::validateInput("vals", prhs[0]);
	double* vals = mxGetPr(prhs[0]);
	mwSize nnz = mxGetNumberOfElements(prhs[0]);

	// col_ind: column indices
	ErrorHandling::validateInput("col_ind", prhs[1]);
	ErrorHandling::validateArraySize("col_ind", mxGetNumberOfElements(prhs[1]), nnz);
	double* col_ind_double = mxGetPr(prhs[1]);

	// row_ptr: row pointers
	ErrorHandling::validateInput("row_ptr", prhs[2]);
	double* row_ptr_double = mxGetPr(prhs[2]);
	mwSize M = mxGetNumberOfElements(prhs[2]) - 1;
	mwSize N = M;

	mexPrintf("A matrix (symmetric, lower triangular): M=%lu  N=%lu  nonzero=%lu\n", 
		(unsigned long)M, (unsigned long)N, (unsigned long)nnz);

	// Validate and extract vector b
	ErrorHandling::validateInput("b", prhs[3]);
	ErrorHandling::validateDimensions("b", prhs[3], M, 1);
	double* b = mxGetPr(prhs[3]);

	// Validate and extract tolerance
	ErrorHandling::validateInput("tol", prhs[4], true, true);
	double tol = mxGetScalar(prhs[4]);
	if (tol <= 0.0) {
		ErrorHandling::throwError("ICCG:invalidInput", "Tolerance 'tol' must be positive");
	}

	// Validate and extract maximum iterations
	ErrorHandling::validateInput("max_iter", prhs[5], true, true);
	int max_iter = SafeTypeConversion::toPositiveInt(prhs[5], "max_iter");

	// Validate and extract shift parameter
	ErrorHandling::validateInput("shift", prhs[6], true, true);
	double shift = mxGetScalar(prhs[6]);
	if (shift < 0.0) {
		ErrorHandling::throwError("ICCG:invalidInput", "Shift parameter must be non-negative");
	}

	// Validate and extract initial guess x0
	double* x0 = nullptr;
	if (!mxIsEmpty(prhs[7])) {
		ErrorHandling::validateInput("x0", prhs[7]);
		ErrorHandling::validateDimensions("x0", prhs[7], M, 1);
		x0 = mxGetPr(prhs[7]);
	}

	// Validate and extract options
	Options options;
	if (!mxIsEmpty(prhs[8])) {
		if (!mxIsStruct(prhs[8])) {
			ErrorHandling::throwError("ICCG:invalidInput", "Parameter 'options' must be a struct");
		}
		
		// Validate and extract diverge_factor
		mxArray* diverge_factor_field = mxGetField(prhs[8], 0, "diverge_factor");
		if (diverge_factor_field != nullptr) {
			ErrorHandling::validateInput("options.diverge_factor", diverge_factor_field, true, true);
			options.diverge_factor = mxGetScalar(diverge_factor_field);
			if (options.diverge_factor <= 1.0) {
				ErrorHandling::throwError("ICCG:invalidInput", "options.diverge_factor must be greater than 1.0");
			}
		}
		
		// Validate and extract diverge_count
		mxArray* diverge_count_field = mxGetField(prhs[8], 0, "diverge_count");
		if (diverge_count_field != nullptr) {
			options.diverge_count = SafeTypeConversion::toPositiveInt(diverge_count_field, "options.diverge_count");
		}
		
		// Validate and extract scaling option
		mxArray* scaling_field = mxGetField(prhs[8], 0, "scaling");
		if (scaling_field != nullptr) {
			ErrorHandling::validateInput("options.scaling", scaling_field, true, true);
			if (mxIsLogical(scaling_field)) {
				options.use_scaling = mxIsLogicalScalarTrue(scaling_field);
			} else {
				double scaling_type = mxGetScalar(scaling_field);
				options.use_scaling = (scaling_type != 0.0);
			}
		}
		
		// Validate and extract max_shift_trials
		mxArray* max_shift_trials_field = mxGetField(prhs[8], 0, "max_shift_trials");
		if (max_shift_trials_field != nullptr) {
			options.max_shift_trials = SafeTypeConversion::toPositiveInt(max_shift_trials_field, "options.max_shift_trials");
		}
		
		// Validate and extract shift_increment
		mxArray* shift_increment_field = mxGetField(prhs[8], 0, "shift_increment");
		if (shift_increment_field != nullptr) {
			ErrorHandling::validateInput("options.shift_increment", shift_increment_field, true, true);
			options.shift_increment = mxGetScalar(shift_increment_field);
			if (options.shift_increment <= 0.0) {
				ErrorHandling::throwError("ICCG:invalidInput", "options.shift_increment must be positive");
			}
		}
		
		// Validate and extract max_shift_value
		mxArray* max_shift_value_field = mxGetField(prhs[8], 0, "max_shift_value");
		if (max_shift_value_field != nullptr) {
			ErrorHandling::validateInput("options.max_shift_value", max_shift_value_field, true, true);
			options.max_shift_value = mxGetScalar(max_shift_value_field);
			if (options.max_shift_value <= 0.0) {
				ErrorHandling::throwError("ICCG:invalidInput", "options.max_shift_value must be positive");
			}
		}
		
		// Validate and extract min_diagonal_threshold
		mxArray* min_diagonal_threshold_field = mxGetField(prhs[8], 0, "min_diagonal_threshold");
		if (min_diagonal_threshold_field != nullptr) {
			ErrorHandling::validateInput("options.min_diagonal_threshold", min_diagonal_threshold_field, true, true);
			options.min_diagonal_threshold = mxGetScalar(min_diagonal_threshold_field);
			if (options.min_diagonal_threshold <= 0.0) {
				ErrorHandling::throwError("ICCG:invalidInput", "options.min_diagonal_threshold must be positive");
			}
		}
		
		// Validate and extract zero_diagonal_replacement
		mxArray* zero_diagonal_replacement_field = mxGetField(prhs[8], 0, "zero_diagonal_replacement");
		if (zero_diagonal_replacement_field != nullptr) {
			ErrorHandling::validateInput("options.zero_diagonal_replacement", zero_diagonal_replacement_field, true, true);
			options.zero_diagonal_replacement = mxGetScalar(zero_diagonal_replacement_field);
			if (options.zero_diagonal_replacement <= 0.0) {
				ErrorHandling::throwError("ICCG:invalidInput", "options.zero_diagonal_replacement must be positive");
			}
		}
	}

	// Initialize Incomplete Cholesky Conjugate Gradient solver
	ICCGSolver solver;
	solver.setCurrentShift(shift);

	mexPrintf("tol=%.2g  max_iter=%d  initial_shift=%.3f\n", tol, max_iter, shift);
	mexPrintf("diverge_factor=%.2f  diverge_count=%d  scaling=%s\n", options.diverge_factor, options.diverge_count, options.use_scaling ? "true" : "false");
	mexPrintf("max_shift_trials=%d  shift_increment=%.3f  max_shift_value=%.1f\n", options.max_shift_trials, options.shift_increment, options.max_shift_value);
	mexPrintf("min_diagonal_threshold=%.2g  zero_diagonal_replacement=%.2g\n", options.min_diagonal_threshold, options.zero_diagonal_replacement);

	// Prepare outputs
	plhs[0] = mxCreateDoubleMatrix(M, 1, mxREAL);
	double* x = mxGetPr(plhs[0]);

	// Set initial values
	if (x0) {
		vec_copy(x0, x, M);
	} else {
		std::fill(x, x + M, 0.0);
	}

	// Prepare residual history array
	std::vector<double> residual_log(max_iter + 1);

	// Prepare compressed row storage format structure for input matrix
	SparseMatrix A;
	A.val.assign(vals, vals + nnz);  // Copy from input array to vector
	A.col_idx.resize(nnz);
	A.row_ptr.resize(M + 1);
	A.M = M;
	A.N = N;
	A.nnz = nnz;

	// Index conversion
	bool is_one_based = (row_ptr_double[0] == 1.0);

	// Safe index conversion with comprehensive validation
	SafeTypeConversion::convertIndicesArray(col_ind_double, A.col_idx, nnz, is_one_based, "col_ind");
	SafeTypeConversion::convertIndicesArray(row_ptr_double, A.row_ptr, M + 1, is_one_based, "row_ptr");

	// Validate CSR format consistency
	if (A.row_ptr[0] != 0) {
		ErrorHandling::throwError("ICCG:invalidCSRFormat", 
			"Invalid CSR format: first row pointer must be 0 after index conversion");
	}
	if (A.row_ptr[M] != nnz) {
		ErrorHandling::throwError("ICCG:invalidCSRFormat", 
			"Invalid CSR format: last row pointer must equal total number of non-zeros");
	}
	
	// Validate lower triangular format
	for (mwSize i = 0; i < M; i++) {
		for (mwIndex k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
			if (A.col_idx[k] > i) {
				ErrorHandling::throwError("ICCG:invalidMatrixFormat", 
					"Matrix must contain only lower triangular part (column index <= row index)");
			}
		}
	}

	// Execute Incomplete Cholesky Conjugate Gradient method
	double residual_norm;
	int iterations;
	int flag;
	int iter_best;  // Iteration count of best solution
	incomplete_cholesky_conjugate_gradient(A, b, x, tol, max_iter, solver, options, &residual_norm, &iterations, residual_log.data(), &flag, &iter_best);

	mexPrintf("Final shift parameter: %.3f\n", solver.getCurrentShift());

	// Additional outputs
	if (nlhs >= 2) {
		plhs[1] = mxCreateDoubleScalar(SafeTypeConversion::toDouble(flag));
	}
	if (nlhs >= 3) {
		plhs[2] = mxCreateDoubleScalar(residual_norm);
	}
	if (nlhs >= 4) {
		plhs[3] = mxCreateDoubleScalar(SafeTypeConversion::toDouble(iter_best));  // Return iteration count of best solution
	}
	if (nlhs >= 5) {
		plhs[4] = mxCreateDoubleMatrix(iterations + 1, 1, mxREAL);
		double* residual_out = mxGetPr(plhs[4]);
		// Output residuals for all iterations (including divergence cases)
		for (int i = 0; i <= iterations; i++) {
			residual_out[i] = residual_log[i];
		}
	}
	if (nlhs >= 6) {
		plhs[5] = mxCreateDoubleScalar(solver.getCurrentShift());
	}

	// Memory is automatically freed by std::vector destructors
}