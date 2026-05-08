case_name = 'linear_array_subelement_delay_spatial_impulse';
repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
init_fieldii(repo_root);

fs = 100e6;
c = 1540;
elements = 4;
width = 0.3e-3;
height = 5e-3;
kerf = 0.03e-3;
sub_x = 1;
sub_y = 5;
focus = [0 0 40e-3];
delays = repmat([0 1e-8 2e-8 1e-8 0], elements, 1);
points = [
    0 0 30e-3
    0 0 40e-3
    2e-3 0 40e-3
];

set_field('fs', fs);
set_field('c', c);

aperture = xdc_linear_array(elements, width, height, kerf, sub_x, sub_y, focus);
ele_delay(aperture, (1:elements)', delays);
[samples, start_time] = calc_h(aperture, points);

output_dir = fullfile(repo_root, 'validation', 'results', case_name, 'fieldii');
write_time_response(output_dir, case_name, 'fieldii', samples, start_time, fs);

xdc_free(aperture);
field_end;
