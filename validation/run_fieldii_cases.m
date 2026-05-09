%RUN_FIELDII_CASES Generate all Field II validation reference outputs.

repo_root = fileparts(fileparts(mfilename('fullpath')));
case_root = fullfile(repo_root, 'validation', 'cases');

cases = {
    'single_rectangle_spatial_impulse'
    'piston_spatial_impulse'
    'linear_array_spatial_impulse'
    'focused_linear_array_spatial_impulse'
    'linear_multirow_array_spatial_impulse'
    'focused_multirow_array_spatial_impulse'
    'convex_array_spatial_impulse'
    'convex_focused_array_spatial_impulse'
    'convex_focused_multirow_array_spatial_impulse'
    'two_dimensional_array_spatial_impulse'
    'concave_piston_spatial_impulse'
    'rectangle_aperture_spatial_impulse'
    'triangle_aperture_spatial_impulse'
    'line_bounded_aperture_spatial_impulse'
    'linear_array_emitted_pressure'
    'linear_array_element_waveforms'
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
    'focused_linear_array_geometry'
    'focused_multirow_array_geometry'
    'convex_array_geometry'
    'convex_focused_array_geometry'
    'convex_focused_multirow_array_geometry'
    'concave_piston_geometry'
    'two_dimensional_array_geometry'
    'rectangle_aperture_geometry'
};

for i = 1:numel(cases)
    script = fullfile(case_root, cases{i}, 'fieldii.m');
    fprintf('Running Field II validation case: %s\n', cases{i});
    run(script);
end
