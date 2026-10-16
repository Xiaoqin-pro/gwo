function value = CEC2017Function(x,functionID)
%CEC2017Function  Thin CEC2017 wrapper (row vector in, scalar out).
%   Requires cec17_func.mexw64 and the input_data folder to be reachable from
%   the CURRENT WORKING DIRECTORY, because the mex opens "input_data/...".
    value=cec17_func(x',functionID);
    value=value(1);
end
