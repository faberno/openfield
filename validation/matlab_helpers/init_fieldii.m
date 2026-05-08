function init_fieldii(repo_root)
%INIT_FIELDII Add Field II to the MATLAB path and initialize it.

fieldii_path = getenv('FIELDII_PATH');
if isempty(fieldii_path)
    fieldii_path = fullfile(repo_root, 'supplementary', 'field2', 'Field_II_ver_3_30_windows(1)');
end

if ~exist(fieldii_path, 'dir')
    error('Field II path does not exist: %s', fieldii_path);
end

addpath(fieldii_path);
field_init(-1);
end
