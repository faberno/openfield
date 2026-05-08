case_name = 'piston_spatial_impulse';
repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
init_fieldii(repo_root);

fs = 100e6;
c = 1540;
radius = 5e-3;
element_size = 0.5e-3;
points = [
    0 0 30e-3
    0 0 40e-3
];

set_field('fs', fs);
set_field('c', c);

aperture = xdc_piston(radius, element_size);
[samples, start_time] = calc_h(aperture, points);

output_dir = fullfile(repo_root, 'validation', 'results', case_name, 'fieldii');
write_time_response(output_dir, case_name, 'fieldii', samples, start_time, fs);

xdc_free(aperture);
field_end;
