function write_geometry(output_dir, case_name, source, aperture)
%WRITE_GEOMETRY Write Field II rectangular aperture geometry in validation format.

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

rect = xdc_get(aperture, 'rect');
if size(rect, 1) <= 26 && size(rect, 2) > size(rect, 1)
    rect = rect';
end
physical_indices = rect(:, 1);
if min(physical_indices) >= 1
    physical_indices = physical_indices - 1;
end
subelement_indices = (0:size(rect, 1)-1)';
subelement_centers = rect(:, 17:19);
areas = rect(:, 15) .* rect(:, 16);
normals = rect_normals(rect);
centers = physical_centers(subelement_centers, areas, physical_indices);

writematrix(centers, fullfile(output_dir, 'centers.csv'));
writematrix(subelement_centers, fullfile(output_dir, 'subelement_centers.csv'));
writematrix(normals, fullfile(output_dir, 'normals.csv'));
writematrix(areas, fullfile(output_dir, 'areas.csv'));
writematrix(physical_indices, fullfile(output_dir, 'physical_indices.csv'));
writematrix(subelement_indices, fullfile(output_dir, 'subelement_indices.csv'));

metadata = struct();
metadata.case = case_name;
metadata.source = source;
metadata.kind = 'geometry';
metadata.physical_element_count = max(physical_indices) + 1;
metadata.subelement_count = size(rect, 1);

write_json(fullfile(output_dir, 'metadata.json'), metadata);
end

function normals = rect_normals(rect)
v1 = rect(:, 2:4);
v2 = rect(:, 5:7);
v3 = rect(:, 8:10);
raw = cross(v2 - v1, v3 - v1, 2);
norms = sqrt(sum(raw.^2, 2));
normals = raw ./ norms;
end

function centers = physical_centers(subelement_centers, areas, physical_indices)
count = max(physical_indices) + 1;
centers = zeros(count, 3);
weights = zeros(count, 1);
for i = 1:size(subelement_centers, 1)
    idx = physical_indices(i) + 1;
    centers(idx, :) = centers(idx, :) + subelement_centers(i, :) * areas(i);
    weights(idx) = weights(idx) + areas(i);
end
centers = centers ./ weights;
end
