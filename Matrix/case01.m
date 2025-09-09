clear all;
close all;
format long;
warning off;


rng(0);
N = 10000;
A = sprandsym(N, 10/N) + 7.4*speye(N);
b = ones(N,1);

save('../Ab.mat','A','b');

