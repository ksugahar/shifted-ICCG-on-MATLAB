function [x, flag, relres, iter, residual_log, shift_used] = iccg_mex(vals, col_ind, row_ptr, b, tol, max_iter, shift, x0, options)
%ICCG_MEX - Incomplete Cholesky Conjugate Gradient Method for Sparse Linear Systems
%
% ## Overview
% Solves sparse symmetric positive definite linear systems Ax = b using the 
% Incomplete Cholesky Conjugate Gradient (ICCG) method with optional scaling.
%
% ## Syntax
% ```matlab
% [x, flag, relres, iter, residual_log, shift_used] = iccg_mex(vals, col_ind, row_ptr, b, tol, max_iter, shift, x0, options)
% ```
%
% ## Input Arguments
% - **vals** - Array of non-zero values (lower triangular part only)
% - **col_ind** - Column index array (1-based or 0-based)
% - **row_ptr** - Row pointer array (1-based or 0-based)
% - **b** - Right-hand side vector (double precision column vector)
% - **tol** - Convergence tolerance (double precision scalar)
% - **max_iter** - Maximum number of iterations (double precision scalar)
% - **shift** - Initial shift parameter for incomplete Cholesky decomposition (double precision scalar)
% - **x0** - Initial guess vector (double precision column vector, or empty array [])
% - **options** - Optional parameters structure:
%   - `.diverge_factor` - Divergence detection factor (default: 10.0)
%   - `.diverge_count` - Divergence detection count (default: 10)
%   - `.scaling` - Enable/disable scaling (logical or numeric, default: true)
%
% ## Output Arguments
% - **x** - Solution vector (double precision column vector)
% - **flag** - Convergence flag:
%   - 0 = Normal convergence
%   - 1 = Maximum iterations reached
%   - 2 = Incomplete Cholesky decomposition failed
%   - 3 = Divergence detected
% - **relres** - Final relative residual
% - **iter** - Iteration number where best solution was found
% - **residual_log** - Residual history for all iterations (including iteration 0)
% - **shift_used** - Final shift parameter used
%
% ## Notes
% - Input matrix A must be symmetric positive definite
% - Only provide the lower triangular part of A
% - Supports both 1-based (MATLAB format) and 0-based (C format) indexing (auto-detected)
% - Implements original algorithm with shift adjustment and scaling capabilities
% - Requires Compressed Sparse Row (CSR) format input
%
% ## Example Usage
% ```matlab
% % Create symmetric sparse matrix (lower triangular part)
% A = sparse([4 -1 0; -1 4 -1; 0 -1 2]);
% b = [1; 2; 3];
% 
% % Convert to CSR format (lower triangular part only)
% [I, J, V] = find(tril(A));
% vals = V;
% col_ind = J;
% row_ptr = accumarray(I, 1, [size(A,1) 1]);
% row_ptr = [0; cumsum(row_ptr)];
% 
% % Solve using ICCG method
% options.scaling = true;
% options.diverge_factor = 10.0;
% [x, flag, relres, iter] = iccg_mex(vals, col_ind, row_ptr, b, 1e-6, 100, 1.0, [], options);
% 
% % Check results
% if flag == 0
%     fprintf('Converged: iterations=%d, relative residual=%.2e\n', iter, relres);
% else
%     fprintf('Did not converge: flag=%d\n', flag);
% end
% ```
%
% ## Algorithm Details
% This MEX function faithfully reproduces the original ICCG algorithm with the following features:
% - IC(0) incomplete Cholesky decomposition preconditioning
% - Automatic shift parameter adjustment
% - Optional scaling functionality
% - Divergence detection capability
% - Best solution preservation and restoration
%
% ## See Also
% PCG, ICHOL, SPARSE, SPDIAGS

% This function is a MEX file - implementation is in iccg_mex.cpp
error('iccg_mex: MEX file not found. Please compile iccg_mex.cpp first.');

end