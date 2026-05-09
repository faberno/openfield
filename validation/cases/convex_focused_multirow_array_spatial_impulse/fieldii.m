case_name = 'convex_focused_multirow_array_spatial_impulse';
repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
run_aperture_calc_h_case(repo_root, case_name);
