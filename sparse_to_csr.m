function [vals, col_ind, row_ptr] = sparse_to_csr(A)
	% MATLABのsparse行列をCSR形式に変換

	[m, n] = size(A);
	[i, j, v] = find(A);  % COO形式で取得

	% 行でソート
	[i_sorted, idx] = sort(i);
	j_sorted = j(idx);
	v_sorted = v(idx);

	% CSR形式の構築
	vals = v_sorted;
	col_ind = j_sorted - 1;  % 0-indexed

	% row_ptrの構築
	row_ptr = zeros(m + 1, 1);
	for k = 1:length(i_sorted)
		row_ptr(i_sorted(k) + 1) = row_ptr(i_sorted(k) + 1) + 1;
	end

	% 累積和
	row_ptr = cumsum(row_ptr);
end
