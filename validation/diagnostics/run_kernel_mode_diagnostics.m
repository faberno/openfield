function run_kernel_mode_diagnostics(repo_root, isolate_all)
%RUN_KERNEL_MODE_DIAGNOSTICS Generate Field II outputs for kernel switches.

if nargin < 1
    repo_root = fileparts(fileparts(fileparts(mfilename('fullpath'))));
end
if nargin < 2
    isolate_all = false;
end

addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));
init_fieldii(repo_root);
cleanup = onCleanup(@() field_end);

c = 1540;
points = [
    0 0 30e-3
    0 0 40e-3
    2e-3 0 40e-3
];
focus = [0 0 40e-3];

run_polygon_modes(repo_root, c, points, focus);
run_accurate_time_modes(repo_root, c, points, focus);
run_isolated_curved_rectangles(repo_root, c, points, focus, isolate_all);
end


function run_polygon_modes(repo_root, c, points, focus)
fs_values = [50e6 100e6 200e6 500e6 1000e6 2000e6];
for fast = [0 1]
    for fs = fs_values
        set_field('fs', fs);
        set_field('c', c);
        set_field('fast_integration', fast);

        aperture = make_polygon_aperture('triangle', focus);
        diagnostic_name = sprintf('triangle_fast%d_fs%dMHz', fast, round(fs / 1e6));
        write_calc_h(repo_root, diagnostic_name, aperture, points, fs);
        xdc_free(aperture);

        aperture = make_polygon_aperture('line', focus);
        diagnostic_name = sprintf('line_fast%d_fs%dMHz', fast, round(fs / 1e6));
        write_calc_h(repo_root, diagnostic_name, aperture, points, fs);
        xdc_free(aperture);
    end
end
end


function run_accurate_time_modes(repo_root, c, points, focus)
fs = 100e6;
for accurate = [0 1]
    set_field('fs', fs);
    set_field('c', c);
    set_field('accurate_time_calc', accurate);

    for case_index = 1:3
        case_name = curved_case_name(case_index);
        aperture = make_curved_aperture(case_name, focus);
        diagnostic_name = sprintf('%s_accurate%d', case_name, accurate);
        write_calc_h(repo_root, diagnostic_name, aperture, points, fs);
        xdc_free(aperture);
    end
end
end


function run_isolated_curved_rectangles(repo_root, c, points, focus, isolate_all)
fs = 100e6;
set_field('fs', fs);
set_field('c', c);
set_field('accurate_time_calc', 0);

for case_index = 1:3
    case_name = curved_case_name(case_index);
    aperture = make_curved_aperture(case_name, focus);
    rect = fieldii_rectangles(aperture);
    xdc_free(aperture);

    if isolate_all
        selected = 1:size(rect, 1);
    else
        selected = unique(round([1 size(rect, 1) / 2 size(rect, 1)]));
    end
    for selected_index = 1:length(selected)
        rect_index = selected(selected_index);
        aperture = make_curved_aperture(case_name, focus);
        rect = fieldii_rectangles(aperture);
        apply_single_rect_apodization(aperture, rect, rect_index);
        diagnostic_name = sprintf('%s_isolated_rect%03d', case_name, rect_index);
        output_dir = kernel_output_dir(repo_root, diagnostic_name);
        write_calc_h(repo_root, diagnostic_name, aperture, points, fs);
        writematrix(rect(rect_index, :), fullfile(output_dir, 'rect.csv'));
        xdc_free(aperture);
    end
end
end


function rect = fieldii_rectangles(aperture)
rect = xdc_get(aperture, 'rect');
if size(rect, 2) < 26 && size(rect, 1) >= 26
    rect = rect';
elseif size(rect, 1) <= 26 && size(rect, 2) > size(rect, 1)
    rect = rect';
end
end


function aperture = make_polygon_aperture(kind, focus)
switch kind
    case 'triangle'
        triangles = [
            1  -0.5e-3 -0.25e-3 0   0.5e-3 -0.25e-3 0   0.5e-3 0.25e-3 0   1
            1  -0.5e-3 -0.25e-3 0   0.5e-3 0.25e-3 0   -0.5e-3 0.25e-3 0   1
        ];
        aperture = xdc_triangles(triangles, [0 0 0], focus);
    case 'line'
        lines = [
            1 1 0 1 -0.5e-3 0
            1 1 0 1  0.5e-3 1
            1 1 0 0 -0.25e-3 1
            1 1 0 0  0.25e-3 0
        ];
        aperture = xdc_lines(lines, [0 0 0], focus);
    otherwise
        error('Unknown polygon aperture kind: %s', kind);
end
end


function aperture = make_curved_aperture(case_name, focus)
switch case_name
    case 'convex_focused_array'
        aperture = xdc_convex_focused_array(6, 0.3e-3, 5e-3, 0.03e-3, 25e-3, 20e-3, 1, 4, focus);
    case 'convex_focused_multirow_array'
        aperture = xdc_convex_focused_multirow(4, 0.3e-3, 3, [1.0e-3 1.2e-3 1.0e-3], 0.03e-3, 0.05e-3, 25e-3, 20e-3, 1, 2, focus);
    case 'concave_piston'
        aperture = xdc_concave(5e-3, 30e-3, 0.5e-3);
    otherwise
        error('Unknown curved aperture case: %s', case_name);
end
end


function name = curved_case_name(index)
names = {'convex_focused_array', 'convex_focused_multirow_array', 'concave_piston'};
name = names{index};
end


function apply_single_rect_apodization(aperture, rect, rect_index)
physical_ids = rect(:, 1);
if min(physical_ids) == 0
    physical_ids = physical_ids + 1;
end
physical_count = max(physical_ids);
subelement_count = 0;
for physical_index = 1:physical_count
    subelement_count = max(subelement_count, sum(physical_ids == physical_index));
end

physical_index = physical_ids(rect_index);
local_index = sum(physical_ids(1:rect_index) == physical_index);
apo = zeros(physical_count, subelement_count);
apo(physical_index, local_index) = 1;
ele_apodization(aperture, (1:physical_count)', apo);
end


function write_calc_h(repo_root, diagnostic_name, aperture, points, fs)
[samples, start_time] = calc_h(aperture, points);
write_time_response(kernel_output_dir(repo_root, diagnostic_name), diagnostic_name, 'fieldii', samples, start_time, fs);
end


function output_dir = kernel_output_dir(repo_root, diagnostic_name)
output_dir = fullfile(repo_root, 'validation', 'results', 'kernel_modes', diagnostic_name, 'fieldii');
end
