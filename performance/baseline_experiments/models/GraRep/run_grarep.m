function run_grarep(edges_csv, n_nodes, kstep, dim, out_csv)

T = readmatrix(edges_csv);
if isempty(T)
    error('run_grarep: empty edge list');
end

A = sparse(T(:, 1), T(:, 2), 1, n_nodes, n_nodes);
A = A + A';
A = double(A > 0);
R = GraRep(full(A), kstep, dim);
writematrix(R, out_csv);
end
