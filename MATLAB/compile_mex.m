% Compile the ICCG MEX function with separated C++ files
% This script compiles iccg_mex.cpp along with iccg.cpp

clear mex;  % Clear any existing MEX functions

% Compile MEX function with all source files
fprintf('Compiling ICCG MEX function with separated C++ implementation...\n');
try
    mex -v iccg_mex.cpp iccg.cpp
    fprintf('Compilation successful!\n');
catch ME
    fprintf('Compilation failed:\n');
    fprintf('%s\n', ME.message);
    rethrow(ME);
end

% Check if the MEX file was created
if exist('iccg_mex', 'file') == 3
    fprintf('MEX file created successfully: iccg_mex.%s\n', mexext);
else
    error('MEX file was not created');
end

fprintf('\nYou can now use iccg_mex in MATLAB.\n');
fprintf('Run test_iccg.m to test the implementation.\n');