function run_aperture_calc_h_case(repo_root, case_name)
%RUN_APERTURE_CALC_H_CASE Generate Field II calc_h reference output.

addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
init_fieldii(repo_root);

fs = 100e6;
c = 1540;
focus = [0 0 40e-3];
points = [
    0 0 30e-3
    0 0 40e-3
    2e-3 0 40e-3
];

set_field('fs', fs);
set_field('c', c);

switch case_name
    case 'focused_linear_array_spatial_impulse'
        aperture = xdc_focused_array(6, 0.3e-3, 5e-3, 0.03e-3, 20e-3, 1, 4, focus);
    case 'linear_multirow_array_spatial_impulse'
        aperture = xdc_linear_multirow(4, 0.3e-3, 3, [1.0e-3 1.2e-3 1.0e-3], 0.03e-3, 0.05e-3, 1, 2, focus);
    case 'focused_multirow_array_spatial_impulse'
        aperture = xdc_focused_multirow(4, 0.3e-3, 3, [1.0e-3 1.2e-3 1.0e-3], 0.03e-3, 0.05e-3, 20e-3, 1, 2, focus);
    case 'convex_array_spatial_impulse'
        aperture = xdc_convex_array(6, 0.3e-3, 5e-3, 0.03e-3, 25e-3, 1, 3, focus);
    case 'convex_focused_array_spatial_impulse'
        aperture = xdc_convex_focused_array(6, 0.3e-3, 5e-3, 0.03e-3, 25e-3, 20e-3, 1, 4, focus);
    case 'convex_focused_multirow_array_spatial_impulse'
        aperture = xdc_convex_focused_multirow(4, 0.3e-3, 3, [1.0e-3 1.2e-3 1.0e-3], 0.03e-3, 0.05e-3, 25e-3, 20e-3, 1, 2, focus);
    case 'two_dimensional_array_spatial_impulse'
        enabled = [
            1 0 1
            0 1 0
            1 1 0
            0 1 1
        ];
        aperture = xdc_2d_array(4, 3, 0.3e-3, 0.4e-3, 0.03e-3, 0.04e-3, enabled, 2, 2, focus);
    case 'concave_piston_spatial_impulse'
        aperture = xdc_concave(5e-3, 30e-3, 0.5e-3);
    case 'rectangle_aperture_spatial_impulse'
        rect = [
            1  -0.5e-3 -0.25e-3 0   0 -0.25e-3 0   0 0.25e-3 0   -0.5e-3 0.25e-3 0   1 0.5e-3 0.5e-3 -0.25e-3 0 0
            2   0.1e-3 -0.25e-3 0   0.6e-3 -0.25e-3 0   0.6e-3 0.25e-3 0   0.1e-3 0.25e-3 0   1 0.5e-3 0.5e-3 0.35e-3 0 0
        ];
        centers = [
            -0.25e-3 0 0
            0.35e-3 0 0
        ];
        aperture = xdc_rectangles(rect, centers, focus);
    case 'triangle_aperture_spatial_impulse'
        triangles = [
            1  -0.5e-3 -0.25e-3 0   0.5e-3 -0.25e-3 0   0.5e-3 0.25e-3 0   1
            1  -0.5e-3 -0.25e-3 0   0.5e-3 0.25e-3 0   -0.5e-3 0.25e-3 0   1
        ];
        centers = [0 0 0];
        aperture = xdc_triangles(triangles, centers, focus);
    case 'line_bounded_aperture_spatial_impulse'
        lines = [
            1 1 0 1 -0.5e-3 0
            1 1 0 1  0.5e-3 1
            1 1 0 0 -0.25e-3 1
            1 1 0 0  0.25e-3 0
        ];
        centers = [0 0 0];
        aperture = xdc_lines(lines, centers, focus);
    otherwise
        field_end;
        error('Unknown aperture calc_h case: %s', case_name);
end

[samples, start_time] = calc_h(aperture, points);

output_dir = fullfile(repo_root, 'validation', 'results', case_name, 'fieldii');
write_time_response(output_dir, case_name, 'fieldii', samples, start_time, fs);

xdc_free(aperture);
field_end;
end
