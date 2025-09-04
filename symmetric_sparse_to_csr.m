function [vals, col_ind, row_ptr] = symmetric_sparse_to_crs(A)
	% 対称sparse行列の下三角部分のみをCRS形式に変換
	% 入力: A - 対称sparse行列
	% 出力: vals - 非ゼロ要素の値（下三角部分のみ）
	%       col_ind - 列インデックス（0-indexed）
	%       row_ptr - 各行の開始位置

	[m, n] = size(A);
	
	% 対称性チェック（オプション）
	if m ~= n
		error('入力行列は正方行列である必要があります');
	end
	
	% 下三角部分のみを抽出
	L = tril(A);
	
	[i, j, v] = find(L);  % COO形式で取得（下三角部分のみ）
	
	% 行でソート
	[i_sorted, idx] = sort(i);
	j_sorted = j(idx);
	v_sorted = v(idx);
	
	% CRS形式の構築
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