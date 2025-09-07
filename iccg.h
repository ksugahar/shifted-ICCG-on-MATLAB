#ifndef ICCG_H
#define ICCG_H

#include <vector>
#include <cstddef>

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
	std::vector<size_t> row_ptr;	// Row pointers
	std::vector<size_t> col_idx;	// Column indices
	size_t M;	// Number of rows
	size_t N;	// Number of columns
	size_t nnz;	// Number of non-zero elements
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

// Main ICCG solver function
void incomplete_cholesky_conjugate_gradient(
	const SparseMatrix& A,
	const double* b,
	double* x,
	double tol,
	int max_iter,
	ICCGSolver& solver,
	const Options& options,
	double* residual_norm,
	int* iterations,
	double* residual_log,
	int* flag,
	int* iter_best
);

// Helper functions for incomplete Cholesky decomposition
bool ichol_original_algorithm(
	SparseMatrix& L,
	const SparseMatrix& A,
	ICCGSolver& solver,
	std::vector<double>& scaling,
	std::vector<double>& D_out,
	const Options& options
);

// Apply preconditioning
void apply_preconditioner_original(
	const SparseMatrix& L,
	const std::vector<double>& D,
	const std::vector<double>& scaling,
	const double* r,
	double* z,
	double* temp,
	const Options& options
);

// Vector operations
void vec_copy(const double* src, double* dst, size_t n);
double vec_dot(const double* x, const double* y, size_t n);
void vec_axpy(double alpha, const double* x, double* y, size_t n);
void vec_scale(double alpha, double* x, size_t n);

// Sparse matrix operations
void sparse_symm_matvec(const SparseMatrix& A, const double* x, double* y);
double get_matrix_value(const SparseMatrix& A, size_t row, size_t col);

#endif // ICCG_H