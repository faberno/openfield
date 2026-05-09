case_name = 'concave_piston_geometry';
repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
run_aperture_geometry_case(repo_root, case_name);
