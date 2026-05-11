function run_fieldii_benchmarks(cases, repeats)
%RUN_FIELDII_BENCHMARKS Time Field II validation cases.
%
% Examples:
%   run_fieldii_benchmarks()
%   run_fieldii_benchmarks('all', 1)
%   run_fieldii_benchmarks({'single_rectangle_spatial_impulse'}, 3)

repo_root = fileparts(fileparts(mfilename('fullpath')));
if nargin < 1 || isempty(cases)
    cases = default_benchmark_cases();
elseif ischar(cases) || isstring(cases)
    if strcmp(string(cases), "all")
        cases = all_validation_cases(repo_root);
    else
        cases = cellstr(cases);
    end
end
if nargin < 2 || isempty(repeats)
    repeats = 1;
end
if repeats < 1
    error('repeats must be >= 1');
end

case_root = fullfile(repo_root, 'validation', 'cases');
addpath(fullfile(repo_root, 'validation', 'matlab_helpers'));

for i = 1:numel(cases)
    case_id = cases{i};
    script = fullfile(case_root, case_id, 'fieldii.m');
    if ~exist(script, 'file')
        error('Missing Field II validation script: %s', script);
    end

    fprintf('Benchmarking Field II validation case: %s\n', case_id);
    elapsed_seconds = zeros(repeats, 1);
    for repeat = 1:repeats
        timer = tic;
        run(script);
        elapsed_seconds(repeat) = toc(timer);
        fprintf('  repeat %d/%d: %.6g s\n', repeat, repeats, elapsed_seconds(repeat));
    end
    write_runtime(repo_root, case_id, elapsed_seconds);
end
end

function cases = default_benchmark_cases()
cases = {
    'single_rectangle_spatial_impulse'
    'linear_array_spatial_impulse'
    'triangle_aperture_spatial_impulse'
    'linear_array_emitted_pressure'
    'linear_array_pulse_echo'
    'linear_array_full_matrix_capture'
    'linear_array_geometry'
    'rectangle_aperture_geometry'
};
end

function cases = all_validation_cases(repo_root)
payload = jsondecode(fileread(fullfile(repo_root, 'validation', 'cases.json')));
cases = cell(numel(payload), 1);
for index = 1:numel(payload)
    if iscell(payload)
        entry = payload{index};
    else
        entry = payload(index);
    end
    cases{index} = entry.name;
end
end

function write_runtime(repo_root, case_id, elapsed_seconds)
output_dir = fullfile(repo_root, 'validation', 'results', case_id, 'fieldii');
if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

payload = struct();
payload.case = case_id;
payload.source = 'fieldii';
payload.runtime_scope = 'validation case script, including setup and output writing';
payload.repeats = numel(elapsed_seconds);
payload.elapsed_seconds = elapsed_seconds(:)';
payload.min_seconds = min(elapsed_seconds);
payload.median_seconds = median(elapsed_seconds);
payload.mean_seconds = mean(elapsed_seconds);
payload.matlab_version = version;

write_json(fullfile(output_dir, 'runtime.json'), payload);
end
