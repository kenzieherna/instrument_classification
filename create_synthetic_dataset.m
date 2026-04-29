function [features, labels, instrument_classes] = create_synthetic_dataset(n_per_class, fs, nfft, hop, n_mfcc)
% CREATE_SYNTHETIC_DATASET (without Cello)
%   Generates 2-second multi-note sequences with simulated reverb
%   5 instrument classes: Piano, Violin, Flute, Trumpet, Guitar

    instrument_defs = {
     % name      f0_lo  f0_hi  nH  roll   noise  inharm   env
      'Piano',   500,   1500,   14, 1.20,  0.010, 0.0008, 'piano';
      'Violin',  300,   1200,    8, 1.05,  0.025, 0.0,    'bowed';
      'Flute',   350,   900,     4, 1.90,  0.035, 0.0,    'blown';
      'Trumpet', 200,   700,    15, 0.85,  0.018, 0.0,    'brass';
      'Guitar',  60,    250,     6, 2.10,  0.040, 0.00005, 'plucked';
    };

    n_classes          = size(instrument_defs, 1);
    instrument_classes = instrument_defs(:, 1);

    seg_duration = 2.0;
    n_samples    = round(seg_duration * fs);

    all_features = [];
    all_labels   = [];

    window   = make_window(nfft, 'hann');
    noverlap = nfft - hop;

    fprintf('Generating synthetic data (5 instruments):\n');

    for c = 1:n_classes
        f0_lo     = instrument_defs{c, 2};
        f0_hi     = instrument_defs{c, 3};
        n_harm    = instrument_defs{c, 4};
        rolloff   = instrument_defs{c, 5};
        noise_lvl = instrument_defs{c, 6};
        inharm    = instrument_defs{c, 7};
        env_type  = instrument_defs{c, 8};

        for s = 1:n_per_class
            % Generate a 2-second multi-note sequence
            sig = generate_sequence(env_type, f0_lo, f0_hi, n_harm, ...
                                    rolloff, inharm, noise_lvl, n_samples, fs);

            % Simulate room reverb
            sig = add_reverb(sig, fs, env_type);

            % Add background noise
            bg_noise = 0.005 * randn(1, n_samples);
            sig = sig + bg_noise;

            % Normalize
            sig = sig / (max(abs(sig)) + 1e-8);

            % Extract features
            [S, f_vec] = my_stft(sig, window, noverlap, nfft, fs);
            S_mag = abs(S);

            feat         = extract_features(S_mag, f_vec, sig, fs, nfft, hop, n_mfcc);
            all_features = [all_features; feat];
            all_labels   = [all_labels;   c];
        end

        fprintf('  [%d/%d] %-8s\n', c, n_classes, instrument_defs{c,1});
    end

    features = all_features;
    labels   = all_labels;
    
    fprintf('Complete: %d samples, %d features per sample\n\n', size(features,1), size(features,2));
end


% =========================================================================
function sig = generate_sequence(env_type, f0_lo, f0_hi, n_harm, ...
                                  rolloff, inharm, noise_lvl, n_samples, fs)

    sig = zeros(1, n_samples);
    t   = (0:n_samples-1) / fs;
    f_nyq = fs / 2;

    switch env_type
        case 'plucked'
            % Guitar: fast decay
            n_notes    = 3 + randi(3);
            note_times = sort(rand(1, n_notes)) * 1.6;

            for n = 1:n_notes
                f0_n = exp(log(f0_lo) + rand() * log(f0_hi/f0_lo));
                n_h  = min(n_harm, floor(f_nyq/f0_n));
                if n_h < 1, n_h = 1; end

                onset      = round(note_times(n) * fs) + 1;
                attack     = round(0.020 * fs);
                decay_rate = 3.5 + rand() * 2.5;

                note = zeros(1, n_samples);
                for h = 1:n_h
                    f_h = h * f0_n * sqrt(1 + inharm * h^2);
                    if f_h >= f_nyq, break; end
                    note = note + (1/h^rolloff) * sin(2*pi*f_h*t + 2*pi*rand());
                end

                env = zeros(1, n_samples);
                for i = onset:n_samples
                    i_rel = i - onset;
                    if i_rel < attack
                        env(i) = i_rel / attack;
                    else
                        env(i) = exp(-decay_rate * (i_rel - attack) / fs);
                    end
                end

                % Pick transient
                pick_len = round(0.008 * fs);
                p_end    = min(onset + pick_len - 1, n_samples);
                pick_noise = 0.12 * randn(1, pick_len) .* linspace(1,0,pick_len);
                sig(onset:p_end) = sig(onset:p_end) + pick_noise(1:p_end-onset+1);

                sig = sig + note .* env + noise_lvl * randn(1,n_samples) .* env;
            end

        case {'bowed', 'blown', 'brass'}
            % Violin/Flute/Trumpet: legato, sustained
            n_notes     = 2 + randi(2);
            note_dur    = 1.8 / n_notes;
            crossfade   = round(0.05 * fs);

            for n = 1:n_notes
                f0_n = exp(log(f0_lo) + rand() * log(f0_hi/f0_lo));
                n_h  = min(n_harm, floor(f_nyq/f0_n));
                if n_h < 1, n_h = 1; end

                onset_s = round((n-1) * note_dur * fs) + 1;
                end_s   = min(round(n * note_dur * fs), n_samples);
                dur_s   = end_s - onset_s + 1;

                note = zeros(1, n_samples);
                t_local = (0:dur_s-1) / fs;

                % Vibrato for bowed
                if strcmp(env_type, 'bowed')
                    vib_rate  = 5 + rand() * 1.5;
                    vib_depth = 0.003 + rand() * 0.002;
                    vib = 1 + vib_depth * sin(2*pi*vib_rate*t_local);
                else
                    vib = ones(1, dur_s);
                end

                for h = 1:n_h
                    f_h = h * f0_n;
                    if f_h >= f_nyq, break; end
                    phase_inc = cumsum(2*pi*f_h*vib/fs);
                    note_seg  = (1/h^rolloff) * sin(phase_inc + 2*pi*rand());
                    note(onset_s:end_s) = note(onset_s:end_s) + note_seg;
                end

                % Smooth envelope with crossfade
                env_seg = ones(1, dur_s);
                fade    = min(crossfade, round(dur_s/4));
                env_seg(1:fade)           = linspace(0, 1, fade);
                env_seg(end-fade+1:end)   = linspace(1, 0, fade);

                note(onset_s:end_s) = note(onset_s:end_s) .* env_seg;

                % Instrument-specific noise
                if strcmp(env_type, 'bowed')
                    bow_noise = noise_lvl * 1.5 * randn(1, dur_s) .* env_seg;
                    note(onset_s:end_s) = note(onset_s:end_s) + bow_noise;
                elseif strcmp(env_type, 'blown')
                    breath = noise_lvl * 2.5 * randn(1, dur_s) .* env_seg;
                    note(onset_s:end_s) = note(onset_s:end_s) + breath;
                end

                sig = sig + note;
            end

    case 'piano'
        % Piano: realistic hammer strikes with string resonance
        n_notes    = 2 + randi(2);   % 2-3 notes only
        note_times = sort(rand(1, n_notes)) * 1.5;
    
        for n = 1:n_notes
            % Use WIDER frequency range to match real pianos
            f0 = exp(log(300) + rand() * log(1200/300));  % 300-1200 Hz (wider!)
            n_h  = min(n_harm, floor(f_nyq/f0));
            if n_h < 1, n_h = 1; end
        
            onset = round(note_times(n) * fs) + 1;
        
            % LONGER attack for realistic hammer strike
            attack = round(0.040 * fs);  % 40ms (was 8ms) - realistic hammer
        
            % Stronger initial decay, longer sustain
            decay1 = round(0.25 * fs);   % 250ms (was 120ms)
            sus_level = 0.55 + rand() * 0.20;  % Higher sustain level
            decay2_rate = 0.3 + rand() * 0.2;  % SLOWER decay (was 0.8-1.2)
        
            note = zeros(1, n_samp);
            for h = 1:n_h
                % Add slight inharmonicity (real pianos have this!)
                f_h = h * f0 * sqrt(1 + 0.0005 * h^2);  % inharmonicity factor
                if f_h >= f_nyq, break; end
                note = note + (1/h^rolloff) * sin(2*pi*f_h*t + 2*pi*rand());
            end
        
            env = zeros(1, n_samp);
            for i = onset:n_samp
                i_rel = i - onset;
                if i_rel < attack
                    env(i) = (i_rel / attack)^1.5;  % Smoother attack curve
                elseif i_rel < attack + decay1
                    env(i) = 1 - (1 - sus_level) * (i_rel - attack) / decay1;
                else
                    env(i) = sus_level * exp(-decay2_rate * (i_rel - attack - decay1) / fs);
                end
            end
        
            sig = sig + note .* env + noise_lvl * randn(1, n_samp) .* env;
        end
    
   end
end


% =========================================================================
function sig_out = add_reverb(sig, fs, env_type)

    switch env_type
        case 'plucked'
            rt60 = 0.05 + rand() * 0.10;
            wet  = 0.10 + rand() * 0.15;
        case {'bowed', 'blown', 'brass'}
            rt60 = 0.5  + rand() * 0.8;
            wet  = 0.5  + rand() * 0.3;
        case 'piano'
            rt60 = 0.4  + rand() * 0.5;
            wet  = 0.4  + rand() * 0.3;
        otherwise
            rt60 = 0.3;
            wet  = 0.4;
    end

    ir_len  = max(round(rt60 * fs), 10);
    ir      = randn(1, ir_len) .* exp(-3 * linspace(0, 1, ir_len));
    ir      = ir / (max(abs(ir)) + 1e-8);

    sig_rev = conv(sig, ir);
    sig_out = (1-wet)*sig + wet*sig_rev(1:length(sig));
    sig_out = sig_out / (max(abs(sig_out)) + 1e-8);
end


% =========================================================================
function feat = extract_features(S_mag, f_vec, sig, fs, nfft, hop, n_mfcc)
    n_frames = size(S_mag, 2);
    P        = S_mag .^ 2;

    sc_all   = zeros(1, n_frames);
    sr_all   = zeros(1, n_frames);
    sf_all   = zeros(1, n_frames);
    zcr_all  = zeros(1, n_frames);
    flat_all = zeros(1, n_frames);

    for m = 1:n_frames
        P_m    = P(:, m);
        P_norm = P_m / (sum(P_m) + 1e-10);

        sc_all(m) = sum(f_vec .* P_norm);

        cdf = cumsum(P_norm);
        idx = find(cdf >= 0.85, 1, 'first');
        if isempty(idx), idx = length(f_vec); end
        sr_all(m) = f_vec(idx);

        if m > 1
            sf_all(m) = sum((S_mag(:,m) - S_mag(:,m-1)).^2);
        end

        frame_start = (m-1)*hop + 1;
        frame_end   = min(frame_start + nfft - 1, length(sig));
        frame       = sig(frame_start:frame_end);
        if length(frame) > 1
            zcr_all(m) = sum(frame(1:end-1) .* frame(2:end) < 0) / length(frame);
        end

        gm = exp(mean(log(P_m + 1e-10)));
        am = mean(P_m) + 1e-10;
        flat_all(m) = gm / am;
    end

    rms_frames = sqrt(mean(P, 1));
    edr = compute_energy_decay_rate(rms_frames);
    lat = compute_log_attack_time(sig, fs);

    burst = std(rms_frames) / (mean(rms_frames) + 1e-10);

    P_mean = mean(P, 2);
    total_power = sum(P_mean) + 1e-10;

    bands = [0, 250, 500, 1000, 2000, 4000, f_vec(end)];
    band_ratios = zeros(1, length(bands)-1);
    for b = 1:length(bands)-1
        idx_b = f_vec >= bands(b) & f_vec < bands(b+1);
        band_ratios(b) = sum(P_mean(idx_b)) / total_power;
    end
    low_band_ratio  = sum(P_mean(f_vec < 700))  / total_power;
    high_band_ratio = sum(P_mean(f_vec > 2000)) / total_power;

    mfcc_mat = compute_mfcc(S_mag, f_vec, fs, nfft, n_mfcc, 40);

    agg = @(x) [mean(x), std(x), median(x)];

    feat = [agg(sc_all), agg(sr_all), agg(sf_all), agg(zcr_all), ...
            agg(flat_all), edr, lat, burst, ...
            band_ratios, low_band_ratio, high_band_ratio, ...
            agg_rows(mfcc_mat)];
end


function edr = compute_energy_decay_rate(rms_frames)
    log_rms = log(rms_frames + 1e-10);
    [~, peak_idx] = max(log_rms);
    tail   = log_rms(peak_idx:end);
    n_tail = length(tail);
    if n_tail < 3, edr = 0; return; end
    x     = (1:n_tail)';
    slope = (n_tail*sum(x.*tail(:)) - sum(x)*sum(tail(:))) / ...
            (n_tail*sum(x.^2) - sum(x)^2);
    edr   = slope;
end


function lat = compute_log_attack_time(sig, fs)
    env        = abs(sig);
    win        = round(0.020 * fs);
    env_smooth = conv(env, ones(1,win)/win, 'same');
    peak_val   = max(env_smooth);
    if peak_val < 1e-8, lat = -3; return; end
    onset_idx = find(env_smooth >= 0.1*peak_val, 1, 'first');
    peak_idx  = find(env_smooth >= 0.9*peak_val, 1, 'first');
    if isempty(onset_idx) || isempty(peak_idx) || peak_idx <= onset_idx
        lat = -3; return;
    end
    lat = log10((peak_idx - onset_idx)/fs + 1e-5);
end


function out = agg_rows(M)
    out = [];
    for r = 1:size(M,1)
        row = M(r,:);
        out = [out, mean(row), std(row), median(row)];
    end
end


function mfcc_matrix = compute_mfcc(S_mag, f_vec, fs, nfft, n_mfcc, n_filters)
    if nargin < 6, n_filters = 40; end
    mel  = @(f) 2595 * log10(1 + f / 700);
    imel = @(m) 700 * (10.^(m / 2595) - 1);
    mel_pts = linspace(mel(0), mel(fs/2), n_filters + 2);
    hz_pts  = imel(mel_pts);
    bin_pts = max(1, min(round((nfft+1) * hz_pts / fs), size(S_mag,1)));
    H = zeros(n_filters, size(S_mag,1));
    for m = 1:n_filters
        lo = bin_pts(m); ctr = bin_pts(m+1); hi = bin_pts(m+2);
        for k = lo:ctr
            if ctr > lo, H(m,k) = (k-lo)/(ctr-lo); end
        end
        for k = ctr:hi
            if hi > ctr, H(m,k) = (hi-k)/(hi-ctr); end
        end
    end
    log_mel     = log(H * (S_mag.^2) + 1e-10);
    mfcc_matrix = zeros(n_mfcc, size(S_mag,2));
    for i = 1:n_mfcc
        mfcc_matrix(i,:) = sum(log_mel .* cos(pi*(i-1)/n_filters * ((1:n_filters)'-0.5)), 1);
    end
end