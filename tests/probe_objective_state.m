function probe_objective_state()
%PROBE_OBJECTIVE_STATE  Is cec17_func order-dependent / does it consume the RNG?
root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'),fullfile(root,'data','cec2017'));
cd(fullfile(root,'data','cec2017'));
rng(42,'twister'); P=-100+200*rand(6,50);
ord=[6 1 4 2 5 3];
a=zeros(6,1); for i=1:6, a(i)=CEC2017Function(P(i,:),1); end
b=zeros(6,1); for i=1:6, b(i)=CEC2017Function(P(ord(i),:),1); end
fprintf('order-independent (F1): %d   maxdiff=%.3g\n', isequal(a(ord),b), max(abs(a(ord)-b)));
for fn=[14 21 30]
    c=zeros(6,1); for i=1:6, c(i)=CEC2017Function(P(i,:),fn); end
    d=zeros(6,1); for i=1:6, d(i)=CEC2017Function(P(ord(i),:),fn); end
    fprintf('order-independent (F%d): %d   maxdiff=%.3g\n', fn, isequal(c(ord),d), max(abs(c(ord)-d)));
end
rng(1,'twister'); u1=rand(1,50);
rng(1,'twister'); v=CEC2017Function(P(1,:),14); u2=rand(1,50);
fprintf('mex consumed global RNG: %d   (%.12f vs %.12f)\n',~isequal(u1,u2),u1(1),u2(1));
end
