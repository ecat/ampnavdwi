function [signal_pre_tipup, signal_post_tipup, tipup_phases] = epg_0c_tipup(B1s, N_tipup_phases, ...
    N_refoc_pulses, T1, T2, TE, dwi_attenuation_tau, refocus_flip_angle)

N_B1s = numel(B1s);
tipup_phases = linspace(0, 2 * pi, N_tipup_phases);

signal_post_tipup = cell(N_B1s, N_tipup_phases);
signal_pre_tipup = cell(N_B1s);

FZ_0 = [0; 0; 1];

for bb = 1: N_B1s
    B1 = B1s(bb);
    FZ_1 = epg_rf(FZ_0, B1 * pi/2, 0);

    if(N_refoc_pulses == 1)    
        FZ_2 = epg_relax(FZ_1, T1, T2, TE/2);
        FZ_3 = epg_grad(FZ_2);
        FZ_4 = epg_rf(FZ_3, B1 * refocus_flip_angle, pi/2);
        FZ_5 = epg_grad(FZ_4);
        FZ_6 = epg_relax(FZ_5, T1, T2, TE/2);
        FZ_pre_tipup = FZ_6;
    else
        FZ_2 = epg_grad(epg_relax(FZ_1, T1, T2, TE/4));
        FZ_3 = epg_grad(epg_rf(FZ_2, B1 * refocus_flip_angle, pi/2));
        FZ_4 = epg_grad(epg_relax(FZ_3, T1, T2, TE/2));
        FZ_5 = epg_grad(epg_rf(FZ_4, B1 * refocus_flip_angle, pi/2));
        FZ_6 = epg_relax(FZ_5, T1, T2, TE/4);
        FZ_pre_tipup = FZ_6;
    end

    FZ_pre_tipup = epg_relax(FZ_pre_tipup, 10000, 1, dwi_attenuation_tau);

    signal_pre_tipup{bb} = FZ_pre_tipup(:, 1:2);

    for tt = 1: N_tipup_phases
        theta = tipup_phases(tt);
        FZ_7 = epg_rf(FZ_pre_tipup, -B1 * pi/2, theta);
        signal_post_tipup{bb, tt} = FZ_7(:, 1:2);
    end
end
end

