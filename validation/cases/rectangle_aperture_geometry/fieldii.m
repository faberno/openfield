case_name = 'rectangle_aperture_geometry';
repo_root = fileparts(fileparts(fileparts(fileparts(mfilename('fullpath')))));
addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
init_fieldii(repo_root);

rect = [
    1  -0.5e-3 -0.25e-3 0   0 -0.25e-3 0   0 0.25e-3 0   -0.5e-3 0.25e-3 0   1 0.5e-3 0.5e-3 -0.25e-3 0 0
    2   0.1e-3 -0.25e-3 0   0.6e-3 -0.25e-3 0   0.6e-3 0.25e-3 0   0.1e-3 0.25e-3 0   1 0.5e-3 0.5e-3 0.35e-3 0 0
];
centers = [
    -0.25e-3 0 0
    0.35e-3 0 0
];
focus = [0 0 30e-3];

aperture = xdc_rectangles(rect, centers, focus);

output_dir = fullfile(repo_root, 'validation', 'results', case_name, 'fieldii');
write_geometry(output_dir, case_name, 'fieldii', aperture);

xdc_free(aperture);
field_end;
