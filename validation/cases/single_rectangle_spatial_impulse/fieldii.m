case_name = 'single_rectangle_spatial_impulse';
repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
init_fieldii(repo_root);

fs = 100e6;
c = 1540;
width = 0.3e-3;
height = 0.4e-3;
points = [
    0 0 30e-3
    0.1e-3 0.1e-3 30e-3
    0.4e-3 0 30e-3
];

set_field('fs', fs);
set_field('c', c);

aperture = xdc_linear_array(1, width, height, 0, 1, 1, [0 0 30e-3]);
[samples, start_time] = calc_h(aperture, points);

output_dir = fullfile(repo_root, 'validation', 'results', case_name, 'fieldii');
write_time_response(output_dir, case_name, 'fieldii', samples, start_time, fs);

xdc_free(aperture);
field_end;
