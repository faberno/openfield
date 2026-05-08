function write_json(path, payload)
%WRITE_JSON Write JSON payload to disk.

fid = fopen(path, 'w');
if fid < 0
    error('Could not open %s for writing', path);
end
cleanup = onCleanup(@() fclose(fid));
fprintf(fid, '%s\n', jsonencode(payload));
end
