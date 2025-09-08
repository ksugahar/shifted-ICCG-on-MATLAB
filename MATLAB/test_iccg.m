clear all;
close all;
format long;
warning off;

delete('iccg_mex.mexw64');
if exist('iccg_mex.mexw64', 'file') ~=3
	mex -v COPTIMFLAGS="/O2" COMPFLAGS="$COMPFLAGS /utf-8" -I.. iccg_mex.cpp ..\iccg.cpp;
end

rng(0);
N = 10000;
A = sprandsym(N, 10/N) + 7.4*speye(N);
b = ones(N,1);

set(gcf,'Units','Pixels','Position',[100,30,800,700]);
a1 = axes('Units','Pixels','Position',[120, 85, 600, 600],'FontName','Times New Roman','FontSize',20);
	set(gca,'YScale', 'log');
	hold on;	box on;	grid on; grid minor;

	xlabel('{iteration}');
	ylabel('{residual error}');

label = {};
for scaling = [1.0];
for shift = [1.0, 2.0];
	tol = 1e-9;
	max_iter = 3000;
	options.scaling = scaling;
	options.diverge_factor = 1.03;
	options.diverge_count = 8;

	tic;
	[vals, col_ind, row_ptr] = symmetric_sparse_to_csr(A);
	[x, flag, relres, iter_best, residual_log, shift_used] = iccg_mex(vals, col_ind, row_ptr, b, tol, max_iter, shift, zeros(N,1), options);
	disp(sprintf('mex C		elasped time = %.2f sec', toc));
	disp(sprintf('|x| = %.2f,	|Ax-b|/|b| = %.3g,	iter = %d,	shift = %.2f\n',[norm(x), norm(A*x-b)/norm(b), iter_best, shift_used]));
	plot(residual_log, 'k.-', 'MarkerSize', 18);

	label{end+1} = sprintf('MEX C shift=%.2f',shift_used);
end
end

legend(label, 'box', 'off');
exportgraphics(gcf, sprintf('%s.png', mfilename));

