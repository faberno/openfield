function write_time_response(output_dir, case_name, source, samples, start_time, fs)
%WRITE_TIME_RESPONSE Write a Field II time response in validation format.

if ~exist(output_dir, 'dir')
    mkdir(output_dir);
end

writematrix(samples, fullfile(output_dir, 'samples.csv'));
time = start_time + (0:size(samples, 1)-1)' / fs;
writematrix(time, fullfile(output_dir, 'time.csv'));

metadata = struct();
metadata.case = case_name;
metadata.source = source;
metadata.kind = 'time_response';
metadata.sampling_frequency = fs;
metadata.start_time = start_time;
metadata.samples_shape = size(samples);

write_json(fullfile(output_dir, 'metadata.json'), metadata);
end
