function run_aperture_geometry_case(repo_root, case_name)
%RUN_APERTURE_GEOMETRY_CASE Generate Field II aperture geometry output.

addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
init_fieldii(repo_root);

focus = [0 0 40e-3];

switch case_name
    case 'focused_linear_array_geometry'
        aperture = xdc_focused_array(6, 0.3e-3, 5e-3, 0.03e-3, 20e-3, 1, 4, focus);
    case 'focused_multirow_array_geometry'
        aperture = xdc_focused_multirow(4, 0.3e-3, 3, [1.0e-3 1.2e-3 1.0e-3], 0.03e-3, 0.05e-3, 20e-3, 1, 2, focus);
    case 'convex_array_geometry'
        aperture = xdc_convex_array(6, 0.3e-3, 5e-3, 0.03e-3, 25e-3, 1, 3, focus);
    case 'convex_focused_array_geometry'
        aperture = xdc_convex_focused_array(6, 0.3e-3, 5e-3, 0.03e-3, 25e-3, 20e-3, 1, 4, focus);
    case 'convex_focused_multirow_array_geometry'
        aperture = xdc_convex_focused_multirow(4, 0.3e-3, 3, [1.0e-3 1.2e-3 1.0e-3], 0.03e-3, 0.05e-3, 25e-3, 20e-3, 1, 2, focus);
    case 'concave_piston_geometry'
        aperture = xdc_concave(5e-3, 30e-3, 0.5e-3);
    otherwise
        field_end;
        error('Unknown aperture geometry case: %s', case_name);
end

output_dir = fullfile(repo_root, 'validation', 'results', case_name, 'fieldii');
write_geometry(output_dir, case_name, 'fieldii', aperture);

xdc_free(aperture);
field_end;
end
