function root=setup_project()
%SETUP_PROJECT  Put the frozen implementation and the CEC2017 harness on path.
root=fileparts(mfilename('fullpath'));
addpath(fullfile(root,'src'));
addpath(fullfile(root,'experiments'));
addpath(fullfile(root,'data','cec2017'));
addpath(fullfile(root,'data','cec2017','input_data'));
end
