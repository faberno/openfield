case_name = 'linear_array_geometry';
repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
init_fieldii(repo_root);

elements = 8;
width = 0.3e-3;
height = 5e-3;
kerf = 0.03e-3;
sub_x = 2;
sub_y = 3;
focus = [0 0 40e-3];

aperture = xdc_linear_array(elements, width, height, kerf, sub_x, sub_y, focus);

output_dir = fullfile(repo_root, 'validation', 'results', case_name, 'fieldii');
write_geometry(output_dir, case_name, 'fieldii', aperture);

xdc_free(aperture);
field_end;
