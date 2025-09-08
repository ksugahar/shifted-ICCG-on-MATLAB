#include "mex.h"
#include "matrix.h"
#include "iccg.h"
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
	A.M = static_cast<size_t>(M);
	A.N = static_cast<size_t>(N);
	A.nnz = static_cast<size_t>(nnz);

	// Index conversion
	bool is_one_based = (row_ptr_double[0] == 1.0);

	// Convert indices from double to size_t with validation
	for (mwSize i = 0; i < nnz; i++) {
		double value = is_one_based ? col_ind_double[i] - 1.0 : col_ind_double[i];
		if (value < 0.0 || value > std::numeric_limits<size_t>::max()) {
			ErrorHandling::throwError("ICCG:invalidInput", "Invalid column index");
		}
		A.col_idx[i] = static_cast<size_t>(value);
	}
	
	for (mwSize i = 0; i <= M; i++) {
		double value = is_one_based ? row_ptr_double[i] - 1.0 : row_ptr_double[i];
		if (value < 0.0 || value > std::numeric_limits<size_t>::max()) {
			ErrorHandling::throwError("ICCG:invalidInput", "Invalid row pointer");
		}
		A.row_ptr[i] = static_cast<size_t>(value);
	}

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
	for (size_t i = 0; i < A.M; i++) {
		for (size_t k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
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