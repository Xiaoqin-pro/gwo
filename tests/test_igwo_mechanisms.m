function test_igwo_mechanisms()
%TEST_IGWO_MECHANISMS  Sanity checks for the faithful I-GWO implementation.
%
% Guards the properties the four corrected mechanisms must guarantee, so a later
% edit cannot silently break them:
%   1. exact budget accounting
%   2. T is finite, non-increasing and complete
%   3. the returned solution really has the returned fitness
%   4. personal-best retention makes the run monotone in the budget: a longer
%      run can never end up worse than a shorter one with the same seed
%   5. the returned solution respects the bounds
%
% Run:  setup_project(); test_igwo_mechanisms();

root=fileparts(fileparts(mfilename('fullpath')));
addpath(fullfile(root,'src'),fullfile(root,'data','cec2017'));
cd(fullfile(root,'data','cec2017'));
D=30; N=20; B=2000; seed=77;   % D=30: input_data is shipped for D30/D50 only
rng(20261016,'twister'); pop=-100+200*rand(N,D);
st=struct('initialPopulation',pop,'lowerBound',-100*ones(1,D), ...
          'upperBound',100*ones(1,D),'maxEvaluations',B);
[B1,T1,i1]=IGWO(@(x)CEC2017Function(x,1),st,[],N,seed);
assert(i1.Evaluations==B,sprintf('FE accounting: %d ~= %d',i1.Evaluations,B));
assert(numel(T1)==B && all(isfinite(T1)),'T has holes or is short');
assert(all(diff(T1)<=0),'T is not non-increasing');
assert(all(B1.Vector>=st.lowerBound-1e-9) && all(B1.Vector<=st.upperBound+1e-9), ...
    'returned solution leaves the bounds');
assert(abs(CEC2017Function(B1.Vector,1)-B1.Cost)<1e-9*max(1,abs(B1.Cost)), ...
    'returned fitness does not match the returned solution');
st2=st; st2.maxEvaluations=4000;
[B2,T2,i2]=IGWO(@(x)CEC2017Function(x,1),st2,[],N,seed);
assert(B2.Cost<=B1.Cost+1e-12,'a longer run was worse than a shorter one');
fprintf('test_igwo_mechanisms: all checks passed\n');
fprintf('  budget %d -> %.6g ; budget 4000 -> %.6g\n',B,B1.Cost,B2.Cost);
end
