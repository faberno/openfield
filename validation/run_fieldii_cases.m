%RUN_FIELDII_CASES Generate all Field II validation reference outputs.

repo_root = fileparts(fileparts(mfilename('fullpath')));
case_root = fullfile(repo_root, 'validation', 'cases');

cases = {
    'single_rectangle_spatial_impulse'
    'piston_spatial_impulse'
    'linear_array_spatial_impulse'
    'linear_array_emitted_pressure'
    'linear_array_focus_timeline_spatial_impulse'
    'linear_array_delay_timeline_spatial_impulse'
    'linear_array_apodization_spatial_impulse'
    'linear_array_soft_baffle_spatial_impulse'
    'linear_array_subelement_apodization_spatial_impulse'
    'linear_array_subelement_delay_spatial_impulse'
    'linear_array_pulse_echo'
    'linear_array_scatterer_response'
    'linear_array_receive_channels'
    'linear_array_full_matrix_capture'
    'linear_array_geometry'
    'two_dimensional_array_geometry'
    'rectangle_aperture_geometry'
};

for i = 1:numel(cases)
    script = fullfile(case_root, cases{i}, 'fieldii.m');
    fprintf('Running Field II validation case: %s\n', cases{i});
    run(script);
end
