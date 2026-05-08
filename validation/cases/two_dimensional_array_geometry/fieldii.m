case_name = 'two_dimensional_array_geometry';
repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
init_fieldii(repo_root);

elements_x = 4;
elements_y = 3;
width = 0.3e-3;
height = 0.4e-3;
kerf_x = 0.03e-3;
kerf_y = 0.04e-3;
enabled = [
    1 0 1
    0 1 0
    1 1 0
    0 1 1
];
sub_x = 2;
sub_y = 2;
focus = [0 0 40e-3];

aperture = xdc_2d_array(elements_x, elements_y, width, height, kerf_x, kerf_y, enabled, sub_x, sub_y, focus);

output_dir = fullfile(repo_root, 'validation', 'results', case_name, 'fieldii');
write_geometry(output_dir, case_name, 'fieldii', aperture);

xdc_free(aperture);
field_end;
