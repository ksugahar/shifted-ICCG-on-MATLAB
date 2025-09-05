#include "mex.h"
#include "matrix.h"
#include <cmath>
#include <cstring>
#include <algorithm>
#include <vector>

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

// Options structure (original algorithm compliant)
struct Options {
	double diverge_factor;  // Divergence detection coefficient
	int diverge_count;  // Divergence detection count
	bool use_scaling;  // Control whether to use scaling
	Options() : diverge_factor(10.0), diverge_count(10), use_scaling(true) {}
};

// Compressed row storage format structure for sparse matrices
struct SparseMatrix {
	double* val;	// Values of non-zero elements
	mwIndex* row_ptr;	// Row pointers
	mwIndex* col_idx;	// Column indices
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
	ICCGSolver() : shift(1.0), dShift(0.01), shiftMax(5.0), ifix(false) {}
	
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

// Incomplete Cholesky decomposition that accurately reproduces the original algorithm (lower triangular input version)
bool ichol_original_algorithm(SparseMatrix& L, const SparseMatrix& A, ICCGSolver& solver, std::vector<double>& scaling, std::vector<double>& D_out, const Options& options) {
	mwSize n = A.M;
	double shift = solver.getCurrentShift();
	
	// Adjust initial value of shift parameter
	if (shift < 1.0) {
		shift = 1.0;
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
	L.val = new double[L_nnz];
	L.col_idx = new mwIndex[L_nnz];
	L.row_ptr = new mwIndex[n + 1];
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
	
	// When applying scaling
	if (options.use_scaling) {
		// Calculate scaling
		scaling.resize(n);
		for (mwSize i = 0; i < n; i++) {
			scaling[i] = (original_diag[i] > 0.0) ? 1.0 / sqrt(original_diag[i]) : 1.0;
		}
		// Apply scaling to lower triangular matrix L
		for (mwSize i = 0; i < n; i++) {
			for (mwIndex k = L.row_ptr[i]; k < L.row_ptr[i + 1]; k++) {
				mwIndex j = L.col_idx[k];
				L.val[k] *= scaling[i] * scaling[j];
			}
		}
	} else {
		// If not using scaling, use unit scaling
		scaling.resize(n, 1.0);
	}
	
	// Diagonal element array D (original algorithm doesn't explicitly initialize, but zero initialization is required)
	std::vector<double> D(n, 0.0);
	
	// Save values of input matrix A (for restoration during recalculation)
	std::vector<double> A_original(L_nnz);
	for (mwSize i = 0; i < L_nnz; i++) {
		A_original[i] = L.val[i];
	}
	
	// Also save values after scaling application (for use during recalculation)
	std::vector<double> A_scaled(L_nnz);
	for (mwSize i = 0; i < L_nnz; i++) {
		A_scaled[i] = L.val[i];
	}
	
	// Recalculation processing loop (as per original algorithm)
	bool restart;
	do {
		restart = false;
		
		// Memory copy equivalent: restore lower triangular matrix L with appropriate values
		if (options.use_scaling) {
			// When using scaling, restore with values after scaling application
			for (mwSize i = 0; i < L_nnz; i++) {
				L.val[i] = A_scaled[i];
			}
		} else {
			// When not using scaling, restore with original A matrix values
			for (mwSize i = 0; i < L_nnz; i++) {
				L.val[i] = A_original[i];
			}
		}
		
		mwIndex scol = L.row_ptr[0];
		
		for (mwSize i = 0; i < n; i++) {
			mwIndex ecol = L.row_ptr[i + 1] - 1;
			
			// Process off-diagonal elements
			for (mwIndex k = scol; k < ecol; k++) {
				mwIndex jnf = L.col_idx[k];
				mwIndex jns = L.row_ptr[jnf];
				mwIndex jne = L.row_ptr[jnf + 1] - 1;
				
				// Inner loop of Original algorithm (accurately reproduces boundary conditions)
				for (mwIndex ki = scol, li = jns; li < jne; ) {
					if (ki >= k) break;  // End when index reaches boundary
					
					mwIndex knf = L.col_idx[ki];
					mwIndex lnf = L.col_idx[li];
					
					if (lnf < knf) {
						li++;
					} else if (lnf > knf) {
						ki++;
					} else {
						// Only when lnf < jnf (use only already computed D elements)
						if (lnf < jnf) {
							L.val[k] -= L.val[ki] * L.val[li] / D[lnf];
						}
						ki++;
						li++;
					}
				}
				// Use D[jnf] only when jnf < i (when already computed)
				if (jnf < i) {
					L.val[k] *= D[jnf];
				}
			}
			
			// Process diagonal elements (as per Original implementation)
			double t = options.use_scaling ? A_scaled[ecol] : A_original[ecol];
			double t0 = t;
			if (t > 0.0) t *= shift;
			
			for (mwIndex k = scol; k < ecol; k++) {
				mwIndex col_k = L.col_idx[k];
				// Use only already computed D elements
				if (col_k < i) {
					t -= L.val[k] * L.val[k] / D[col_k];
				}
			}
			
			// Shift adjustment judgment
			if (!solver.isShiftFixed()) {
				if (t < 1.e-6 && t0 > 0.0 && shift < 5.0) {
					restart = true;
					shift += 0.01;
					solver.setCurrentShift(shift);
					break;
				}
			}
			
			// Set D[i]
			if (t <= 0.0) t = 1e-10;
			t = 1.0 / t;
			D[i] = t;
			
			scol = ecol + 1;
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
	mwIndex scol = L.row_ptr[0];
	for (mwSize i = 0; i < n; i++) {
		double t = temp[i];
		mwIndex ecol = L.row_ptr[i + 1] - 1;
		
		for (mwIndex k = scol; k < ecol; k++) {
			t -= L.val[k] * z[L.col_idx[k]];
		}
		z[i] = t;
		scol = ecol + 1;
	}
	
	// Diagonal scaling using D array
	for (mwSize i = 0; i < n; i++) {
		z[i] *= D[i];
	}
	
	// Backward substitution: solve L^T * y = z
	mwIndex ecol = L.row_ptr[n] - 1;
	for (int i = n - 1; i >= 0; i--) {
		mwIndex scol = L.row_ptr[i];
		double t = z[i];
		
		for (mwIndex k = scol; k < L.row_ptr[i + 1] - 1; k++) {
			z[L.col_idx[k]] -= L.val[k] * t;
		}
		ecol = scol - 1;
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
	double* r = new double[n];	// Residual vector
	double* z = new double[n];	// Preconditioned residual
	double* p = new double[n];	// Search direction
	double* Ap = new double[n];	// A * p
	double* temp = new double[n];	// Temporary work
	double* best_x = new double[n];	// Store best solution

	// Maximum number of shift adjustment trials
	int max_shift_trials = 100;
	bool decomposition_success = false;
	
	// L matrix and scaling
	SparseMatrix L_csr;
	std::vector<double> scaling;
	std::vector<double> D;
	
	// Shift adjustment loop
	for (int trial = 0; trial < max_shift_trials; trial++) {
		// Free previous memory (for retry cases)
		if (trial > 0) {
			delete[] L_csr.val;
			delete[] L_csr.col_idx;
			delete[] L_csr.row_ptr;
		}
		
		// Execute IC(0) incomplete Cholesky decomposition (Original algorithm)
		bool success = ichol_original_algorithm(L_csr, A, solver, scaling, D, options);
		
		if (success) {
			decomposition_success = true;
			break;
		} else {
			// Adjust shift and retry
			if (!solver.isShiftFixed() && solver.getCurrentShift() < 5.0) {
				double new_shift = solver.getCurrentShift() + 0.01;
				solver.setCurrentShift(new_shift);
			} else {
				break;  // Adjustment not possible
			}
		}
	}
	
	if (!decomposition_success) {
		*flag = 2;  // Decomposition failed
		delete[] r; delete[] z; delete[] p; delete[] Ap; delete[] temp; delete[] best_x;
		return;
	}

	// Diagonal element array D and scaling array are managed separately

	// Calculate initial residual: r = b - A * x (symmetric matrix version)
	sparse_symm_matvec(A, x, r);
	for (mwSize i = 0; i < n; i++) {
		r[i] = b[i] - r[i];
	}

	// Initial preconditioned residual: z = M^(-1) * r
	apply_preconditioner_original(L_csr, D, scaling, r, z, temp, options);

	// Initial search direction: p = z
	vec_copy(z, p, n);

	// Initial inner products
	double rz_old = vec_dot(r, z, n);
	double rz_new;
	double r0_norm = std::sqrt(vec_dot(r, r, n));
	double r_norm = r0_norm;

	// Divergence detection variables following original algorithm
	double initial_error = r_norm / r0_norm;  // Initial error (following original algorithm)
	double error_minimum;  // Set within convergence judgment function
	vec_copy(x, best_x, n);
	int no_iterations_to_minimum = 0;  // Iteration count where best solution was found
	int flag_store = 0;  // Counter for divergence detection

	// Record residual history (adjusted so that iteration count=1 corresponds to array index=1)
	residual_log[0] = r_norm / r0_norm;  // Initial residual at iteration count 0

	// Incomplete Cholesky Conjugate Gradient iterations (following original algorithm: 1-based counter)
	int iter;
	for (iter = 1; iter <= max_iter; iter++) {
		// Matrix-vector product Ap = A * p (symmetric matrix version)
		sparse_symm_matvec(A, p, Ap);
		
		// Step size alpha = (r' * z) / (p' * A * p)
		double pAp = vec_dot(p, Ap, n);
		double alpha = rz_old / pAp;
		
		// Update solution vector x = x + alpha * p
		vec_axpy(alpha, p, x, n);
		
		// Update residual vector r = r - alpha * A * p
		vec_axpy(-alpha, Ap, r, n);
		
		// Preconditioned residual z = M^(-1) * r
		apply_preconditioner_original(L_csr, D, scaling, r, z, temp, options);
		
		// New inner product
		rz_new = vec_dot(r, z, n);
		
		// Search direction coefficient beta = (r_new' * z_new) / (r_old' * z_old)
		double beta = rz_new / rz_old;
		
		// Update search direction p = z + beta * p
		vec_scale(beta, p, n);
		vec_axpy(1.0, z, p, n);
		
		// Move to next iteration
		rz_old = rz_new;
		r_norm = std::sqrt(vec_dot(r, r, n));
		
		// Record residual history (iteration count=1 corresponds to array index=1)
		residual_log[iter] = r_norm / r0_norm;

		// Divergence detection identical to convergence judgment function from original algorithm
		double current_relative_residual = r_norm / r0_norm;
		
		// Initialization processing for iteration count == 1 (following original algorithm)
		if (iter == 1) {
			flag_store = 0;
			error_minimum = initial_error;
		}
		
		if (current_relative_residual < error_minimum) {
			error_minimum = current_relative_residual;
			vec_copy(x, best_x, n);  // Save best solution
			no_iterations_to_minimum = iter;
			flag_store = 0;  // Reset counter
		} else if (current_relative_residual < error_minimum * options.diverge_factor) {
			flag_store = 0;  // Within tolerance range, so reset counter
		} else {
			++flag_store;  // Outside tolerance range, so increment counter
		}
		
		// Termination condition: (relative error < convergence threshold || divergence flag counter > divergence detection count)
		if (current_relative_residual < tol || flag_store > options.diverge_count) {
			// As in the original algorithm, always restore the best solution upon termination
			vec_copy(best_x, x, n);
			
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
	if (iter > max_iter) {
		vec_copy(best_x, x, n);
		*flag = 1;  // Maximum iteration count reached
	}

	// In Original, only the matrix is scaled, not the solution vector
	// So scaling restoration is not needed

	// Return results
	// Use iter after exiting the loop as is, since it uses iter after the Final() function in Original
	*residual_norm = ((*flag == 3 || *flag == 1) ? error_minimum : r_norm / r0_norm);
	*iterations = iter;  // Actual iteration count (following Original)
	*iter_best = no_iterations_to_minimum;  // Iteration count of best solution
	if (*flag != 3 && *flag != 1) {
		*flag = 0;  // Normal convergence
	}

	// Free memory
	delete[] r;
	delete[] z;
	delete[] p;
	delete[] Ap;
	delete[] temp;
	delete[] best_x;
	delete[] L_csr.val;
	delete[] L_csr.col_idx;
	delete[] L_csr.row_ptr;
}

// MEX gateway function
void mexFunction(int nlhs, mxArray* plhs[], int nrhs, const mxArray* prhs[]) {

	// Check input arguments (vals, col_ind, row_ptr, b, tol, max_iter, shift, x0, options)
	// Note: vals and col_ind contain only the lower triangular part of the symmetric matrix
	if (nrhs != 9) {
		mexErrMsgIdAndTxt("ICCG:invalidNumInputs", 
			"Usage: [x, flag, relres, iter, residual_log, shift_used] = iccg_mex(vals, col_ind, row_ptr, b, tol, max_iter, shift, x0, options)\n"
			"Note: vals and col_ind should contain only the lower triangular part of the symmetric matrix");
	}

	// Check output arguments
	if (nlhs > 6) {
		mexErrMsgIdAndTxt("ICCG:invalidNumOutputs", "Too many output arguments");
	}

	// Get CSR format data for matrix A
	// vals: values of non-zero elements
	if (!mxIsDouble(prhs[0]) || mxIsComplex(prhs[0])) {
		mexErrMsgIdAndTxt("ICCG:invalidInput", "vals must be a real double array");
	}
	double* vals = mxGetPr(prhs[0]);
	mwSize nnz = mxGetNumberOfElements(prhs[0]);

	// col_ind: column indices
	if (!mxIsDouble(prhs[1]) || mxIsComplex(prhs[1])) {
		mexErrMsgIdAndTxt("ICCG:invalidInput", "col_ind must be a real double array");
	}
	if (mxGetNumberOfElements(prhs[1]) != nnz) {
		mexErrMsgIdAndTxt("ICCG:invalidInput", "col_ind must have the same number of elements as vals");
	}
	double* col_ind_double = mxGetPr(prhs[1]);

	// row_ptr: row pointers
	if (!mxIsDouble(prhs[2]) || mxIsComplex(prhs[2])) {
		mexErrMsgIdAndTxt("ICCG:invalidInput", "row_ptr must be a real double array");
	}
	double* row_ptr_double = mxGetPr(prhs[2]);
	mwSize M = mxGetNumberOfElements(prhs[2]) - 1;
	mwSize N = M;

	mexPrintf("A matrix (symmetric, lower triangular): M=%d  N=%d  nonzero=%d\n", M, N, nnz);

	// Get vector b
	if (mxGetM(prhs[3]) != M || mxGetN(prhs[3]) != 1) {
		mexErrMsgIdAndTxt("ICCG:invalidInput", "Vector b must have the same number of rows as the matrix");
	}
	double* b = mxGetPr(prhs[3]);

	// Get tolerance tol
	if (!mxIsScalar(prhs[4])) {
		mexErrMsgIdAndTxt("ICCG:invalidInput", "Tolerance tol must be a scalar");
	}
	double tol = mxGetScalar(prhs[4]);

	// Get maximum iterations max_iter
	if (!mxIsScalar(prhs[5])) {
		mexErrMsgIdAndTxt("ICCG:invalidInput", "Maximum iterations max_iter must be a scalar");
	}
	int max_iter = static_cast<int>(mxGetScalar(prhs[5]));

	// Get shift parameter
	if (!mxIsScalar(prhs[6])) {
		mexErrMsgIdAndTxt("ICCG:invalidInput", "Shift parameter must be a scalar");
	}
	double shift = mxGetScalar(prhs[6]);

	// Get initial guess x0
	double* x0 = nullptr;
	if (!mxIsEmpty(prhs[7])) {
		if (mxGetM(prhs[7]) != M || mxGetN(prhs[7]) != 1) {
			mexErrMsgIdAndTxt("ICCG:invalidInput", "Initial guess x0 must have the same size as b");
		}
		x0 = mxGetPr(prhs[7]);
	}

	// Get options
	Options options;
	if (!mxIsEmpty(prhs[8])) {
		if (!mxIsStruct(prhs[8])) {
			mexErrMsgIdAndTxt("ICCG:invalidInput", "options must be a struct");
		}
		
		// Get diverge_factor
		mxArray* diverge_factor_field = mxGetField(prhs[8], 0, "diverge_factor");
		if (diverge_factor_field != nullptr) {
			if (!mxIsScalar(diverge_factor_field)) {
				mexErrMsgIdAndTxt("ICCG:invalidInput", "options.diverge_factor must be a scalar");
			}
			options.diverge_factor = mxGetScalar(diverge_factor_field);
		}
		
		mxArray* diverge_count_field = mxGetField(prhs[8], 0, "diverge_count");
		if (diverge_count_field != nullptr) {
			if (!mxIsScalar(diverge_count_field)) {
				mexErrMsgIdAndTxt("ICCG:invalidInput", "options.diverge_count must be a scalar");
			}
			options.diverge_count = static_cast<int>(mxGetScalar(diverge_count_field));
		}
		
		// Get scaling (true/false or non-zero/zero)
		mxArray* scaling_field = mxGetField(prhs[8], 0, "scaling");
		if (scaling_field != nullptr) {
			if (!mxIsScalar(scaling_field)) {
				mexErrMsgIdAndTxt("ICCG:invalidInput", "options.scaling must be a scalar");
			}
			if (mxIsLogical(scaling_field)) {
				// For logical values
				options.use_scaling = mxIsLogicalScalarTrue(scaling_field);
			} else {
				// For numeric values: non-zero is true
				double scaling_type = mxGetScalar(scaling_field);
				options.use_scaling = (scaling_type != 0.0);
			}
		}
	}

	// Initialize Incomplete Cholesky Conjugate Gradient solver
	ICCGSolver solver;
	solver.setCurrentShift(shift);

	mexPrintf("tol=%.2g  max_iter=%d  initial_shift=%.3f\n", tol, max_iter, shift);
	mexPrintf("diverge_factor=%.2f  diverge_count=%d  scaling=%s\n", options.diverge_factor, options.diverge_count, options.use_scaling ? "true" : "false");

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
	double* residual_log = new double[max_iter + 1];

	// Prepare compressed row storage format structure for input matrix
	SparseMatrix A;
	A.val = vals;
	A.col_idx = new mwIndex[nnz];
	A.row_ptr = new mwIndex[M + 1];
	A.M = M;
	A.N = N;
	A.nnz = nnz;

	// Index conversion
	bool is_one_based = (row_ptr_double[0] == 1.0);

	if (is_one_based) {
		// Convert from MATLAB's 1-based indices to C++'s 0-based indices
		for (mwSize i = 0; i < nnz; i++) {
			A.col_idx[i] = static_cast<mwIndex>(col_ind_double[i] - 1);
		}
		for (mwSize i = 0; i <= M; i++) {
			A.row_ptr[i] = static_cast<mwIndex>(row_ptr_double[i] - 1);
		}
	} else {
		// Use as is if already 0-based indices
		for (mwSize i = 0; i < nnz; i++) {
			A.col_idx[i] = static_cast<mwIndex>(col_ind_double[i]);
		}
		for (mwSize i = 0; i <= M; i++) {
			A.row_ptr[i] = static_cast<mwIndex>(row_ptr_double[i]);
		}
	}

	// Validity check for compressed row storage format
	if (A.row_ptr[0] != 0) {
		mexErrMsgIdAndTxt("ICCG:invalidCSR", "A.row_ptr[0] must be 0 after conversion");
	}
	if (A.row_ptr[M] != nnz) {
		mexErrMsgIdAndTxt("ICCG:invalidCSR", "A.row_ptr[M] must equal nnz");
	}
	
	// Validity check for lower triangular format
	for (mwSize i = 0; i < M; i++) {
		for (mwIndex k = A.row_ptr[i]; k < A.row_ptr[i + 1]; k++) {
			if (A.col_idx[k] > i) {
				mexErrMsgIdAndTxt("ICCG:invalidInput", 
					"Input matrix must contain only lower triangular part (col_idx <= row_idx)");
			}
		}
	}

	// Execute Incomplete Cholesky Conjugate Gradient method
	double residual_norm;
	int iterations;
	int flag;
	int iter_best;  // Iteration count of best solution
	incomplete_cholesky_conjugate_gradient(A, b, x, tol, max_iter, solver, options, &residual_norm, &iterations, residual_log, &flag, &iter_best);

	mexPrintf("Final shift parameter: %.3f\n", solver.getCurrentShift());

	// Additional outputs
	if (nlhs >= 2) {
		plhs[1] = mxCreateDoubleScalar(static_cast<double>(flag));
	}
	if (nlhs >= 3) {
		plhs[2] = mxCreateDoubleScalar(residual_norm);
	}
	if (nlhs >= 4) {
		plhs[3] = mxCreateDoubleScalar(static_cast<double>(iter_best));  // Return iteration count of best solution
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

	// Free memory
	delete[] A.col_idx;
	delete[] A.row_ptr;
	delete[] residual_log;
}