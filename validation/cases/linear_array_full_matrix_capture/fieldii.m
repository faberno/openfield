case_name = 'linear_array_full_matrix_capture';
repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
init_fieldii(repo_root);

f0 = 3e6;
fs = 100e6;
c = 1540;
elements = 4;
width = 0.3e-3;
height = 5e-3;
kerf = 0.03e-3;
sub_x = 1;
sub_y = 5;
focus = [0 0 40e-3];
points = [
    0 0 30e-3
    0 0 40e-3
    2e-3 0 40e-3
];
amplitudes = [1.0; -0.5; 0.25];
decimation_factor = 1;

set_field('fs', fs);
set_field('c', c);

transmit = xdc_linear_array(elements, width, height, kerf, sub_x, sub_y, focus);
receive = xdc_linear_array(elements, width, height, kerf, sub_x, sub_y, focus);
impulse_response = sin(2*pi*f0*(0:1/fs:2/f0));
impulse_response = impulse_response .* hanning(length(impulse_response))';
excitation = sin(2*pi*f0*(0:1/fs:2/f0));
xdc_impulse(transmit, impulse_response);
xdc_impulse(receive, impulse_response);
xdc_excitation(transmit, excitation);

[samples, start_time] = calc_scat_all(transmit, receive, points, amplitudes, decimation_factor);

output_dir = fullfile(repo_root, 'validation', 'results', case_name, 'fieldii');
write_time_response(output_dir, case_name, 'fieldii', samples, start_time, fs / decimation_factor);

xdc_free(transmit);
xdc_free(receive);
field_end;
