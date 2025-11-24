clc; clear; close all;

% Load audio
[echoed_signal, fs] = audioread('testAudio.mp3');

% Convert to mono if stereo
if size(echoed_signal, 2) > 1
    echoed_signal = mean(echoed_signal, 2);
    fprintf('Converted stereo to mono\n');
end

% Normalize audio
echoed_signal = echoed_signal / max(abs(echoed_signal));

%% Training LMS Block Filter

filter_length = 1024;  
block_size    = 128;
mu            = 1;
num_passes    = 6;     

wts      = zeros(filter_length, 1); 
trainlen = length(echoed_signal);
num_blocks = floor(trainlen / block_size);

fprintf('Processing %d blocks of %d samples each...\n', num_blocks, block_size);
fprintf('Filter length: %d taps\n', filter_length);

% Start timer
tic;  

%% Training Filter
for pass = 1:num_passes
    fprintf('\n=== Training Pass %d/%d ===\n', pass, num_passes);

    for block_idx = 1:num_blocks
        % Get block indices
        start_idx = (block_idx-1)*block_size + 1;
        end_idx   = min(start_idx + block_size - 1, trainlen);
        actual_block_size = end_idx - start_idx + 1;

        % Extract blocks
        d_block = echoed_signal(start_idx:end_idx);

        % Build input matrix - each row is the filter input for one sample
        X = zeros(actual_block_size, filter_length);

        for i = 1:actual_block_size
            sample_idx = start_idx + i - 1;

            % Get the history for this sample
            hist_start = max(1, sample_idx - filter_length + 1);
            hist_end   = sample_idx;
            history    = echoed_signal(hist_start:hist_end);

            % Pad with zeros at the beginning if needed
            if length(history) < filter_length
                history = [zeros(filter_length - length(history), 1); history];
            end

            % Reverse for filter (oldest sample first)
            X(i, :) = flipud(history)';
        end

        % Compute filter output for entire block at once
        y_block = X * wts;

        % Error for entire block
        e_block = d_block - y_block;

        % Block LMS update
        block_power = sum(sum(X.^2)) / actual_block_size + 1e-8;  
        gradient    = (X' * e_block) / actual_block_size;         

        % Weight update
        wts = wts + (mu / block_power) * gradient;

        % Progress
        stride = max(1, floor(num_blocks/10));
        if mod(block_idx, stride) == 0 || block_idx == num_blocks
            mse = mean(e_block.^2);
            wts_norm = norm(wts);
            fprintf('Pass %d - Block %d/%d (%.1f%%), MSE: %.6f, ||w||: %.2f\n', ...
                pass, block_idx, num_blocks, 100*block_idx/num_blocks, mse, wts_norm);
        end
    end
end

training_time = toc;
fprintf('\nTraining completed in %.2f seconds\n', training_time);

%% Applying Filter
fprintf('\n=== Applying Trained Filter ===\n');

tic;
y_cancelled = zeros(trainlen, 1);

for block_idx = 1:num_blocks
    start_idx = (block_idx-1)*block_size + 1;
    end_idx   = min(start_idx + block_size - 1, trainlen);
    actual_block_size = end_idx - start_idx + 1;

    % Build input matrix
    X = zeros(actual_block_size, filter_length);

    for i = 1:actual_block_size
        sample_idx = start_idx + i - 1;

        % Get history for this sample
        hist_start = max(1, sample_idx - filter_length + 1);
        hist_end   = sample_idx;
        history    = echoed_signal(hist_start:hist_end);

        % Pad if needed
        if length(history) < filter_length
            history = [zeros(filter_length - length(history), 1); history];
        end

        % Reverse and assign
        X(i, :) = flipud(history)';
    end

    % Apply filter to block
    y_block = X * wts;
    e_block = echoed_signal(start_idx:end_idx) - y_block;
    y_cancelled(start_idx:end_idx) = e_block;
end

% Process remaining samples
remaining = trainlen - num_blocks * block_size;
if remaining > 0
    for n = num_blocks*block_size+1:trainlen
        hist_start = max(1, n - filter_length + 1);
        history    = echoed_signal(hist_start:n);

        if length(history) < filter_length
            history = [zeros(filter_length - length(history), 1); history];
        end

        y_cancelled(n) = echoed_signal(n) - flipud(history)' * wts;
    end
end

processing_time = toc;
fprintf('Processing completed in %.2f seconds\n', processing_time);
fprintf('Total time: %.2f seconds\n', training_time + processing_time);

%% RESULTS
t = (0:length(echoed_signal)-1)/fs;

figure('Position', [100 100 1200 800]);

subplot(2,1,1)
plot(t, echoed_signal, 'r', 'LineWidth', 1)
title('Echoed Audio')
xlabel('Time (s)'); ylabel('Amplitude')
grid on; xlim([0 min(2, max(t))])

subplot(2,1,2)
plot(t, y_cancelled, 'g', 'LineWidth', 1)
title('Echo-Cancelled Audio')
xlabel('Time (s)'); ylabel('Amplitude')
grid on; xlim([0 min(2, max(t))])

%% Performance Metrics
fprintf('\n=== Performance Metrics ===\n');
erle = 10*log10(mean(echoed_signal.^2) / mean(y_cancelled.^2));
fprintf('Echo Return Loss Enhancement (ERLE): %.2f dB\n', erle);

% Compare segments
segment_len = min(round(1*fs), trainlen);
mid_start   = max(1, min(trainlen - segment_len + 1, round(trainlen/2)));

erle_start  = 10*log10(mean(echoed_signal(1:segment_len).^2) / ...
                       mean(y_cancelled(1:segment_len).^2));

erle_mid    = 10*log10(mean(echoed_signal(mid_start:mid_start+segment_len-1).^2) / ...
                       mean(y_cancelled(mid_start:mid_start+segment_len-1).^2));

erle_end    = 10*log10(mean(echoed_signal(end-segment_len+1:end).^2) / ...
                       mean(y_cancelled(end-segment_len+1:end).^2));

fprintf('ERLE at start: %.2f dB\n', erle_start);
fprintf('ERLE at middle: %.2f dB\n', erle_mid);
fprintf('ERLE at end: %.2f dB\n', erle_end);

%% Save
audiowrite('cancelled_lms.wav', y_cancelled/max(abs(y_cancelled)+eps), fs);
fprintf('\nSaved: cancelled_lms.wav\n');

%% Play
fprintf('\nPlaying echo-cancelled audio...\n');
sound(y_cancelled/max(abs(y_cancelled)), fs);
