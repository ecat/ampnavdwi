%!git clone https://github.com/mribri999/MRSignalsSeqs
addpath MRSignalsSeqs/Matlab
set(groot, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'DefaultLineLineWidth', 1.);
%%
 b_value = 000;
 tissue_type = 3;
 N_refoc_pulses = 2;

TE = 65;
letter_key = int32(tissue_type + b_value);
refocus_flip_angle = 1. * pi;

if(tissue_type == 1)
    tag =  'White Matter';
    T2 = 80;
    T1 = 900;
    D = .8e-3;
elseif(tissue_type == 2)
    tag = 'Gray Matter';
    T2 = 100;
    T1 = 1400;
    D = 1e-3;
elseif(tissue_type == 3)
    tag = 'CSF';
    T2 = 2000;
    T1 = 4000;
    D = 3e-3;
elseif(tissue_type == 4)
    tag = 'Phantom';
    T2 = 500;
    T1 = 2000;
    D = 1.5e-3;
end

if(N_refoc_pulses == 1)
    refoc_pulse_tag = 'Single Refocusing';
else
    refoc_pulse_tag = 'Twice Refocusing';
end

dwi_attenuation_tau = D * b_value;

parameters_tag = sprintf('b = %d $\\mathrm{s}/\\mathrm{mm}^2$', b_value);
left_label =  sprintf('T1 %d ms T2 %d ms MD %.1e $\\mathrm{mm}^2/\\mathrm{s}$', T1, T2, D);
tissue_label = sprintf('%s %s', tag, refoc_pulse_tag);

N_tipup_phases = 180;
%B1s = linspace(0.6, 1.2, N_B1s);
B1s = 0.6:0.01:1.2;
N_B1s = numel(B1s);

[signal_pre_tipup, signal_post_tipup, tipup_phases] = epg_0c_tipup(B1s, ...
    N_tipup_phases, N_refoc_pulses, T1, T2, TE, dwi_attenuation_tau, refocus_flip_angle);

letters = containers.Map('KeyType', 'int32', 'ValueType', 'any'); % keys are tissue type + bvalue
letters(2) = 'A)';
letters(1002) = 'B)';
letters(3) = 'C)';
letters(1003) = 'D)';

do_left_label = (letter_key == 2) || (letter_key == 3);
skip_b_value_label = (letter_key == 3) || (letter_key == 1003);

%%
FZ_pre_tipup = zeros(N_B1s, 2);
Z_post_tipup = zeros(N_B1s, N_tipup_phases, 1);
for bb = 1:N_B1s
    FZ_pre_tipup(bb, 1) = abs(signal_pre_tipup{bb, 1}(1, 1));
    FZ_pre_tipup(bb, 2) = real(signal_pre_tipup{bb, 1}(3, 1));

    for tt = 1:N_tipup_phases
        Z_post_tipup(bb, tt, 1) = real(signal_post_tipup{bb, tt, 1}(3, 1));
    end
end
plot_horizontal_offset = .1;

fig_b1_tissue_sims = figure('Position', [100 100 1000 800], 'Color', 'white'); subplot(211);
plot(B1s, FZ_pre_tipup, 'LineWidth', 2); hold on;
xlabel('B1', 'FontSize', 16);
l = legend('$\mathrm{F}_0$', '$\mathrm{Z}_0$', 'Location', 'south east', 'Interpreter', 'latex');
l.AutoUpdate = 'off';
ylim([-.4, 1.]);
grid on;
ylabel({'Components', 'Pre Tipup'}, 'FontSize', 16);
p = plot([B1s(1) B1s(end)], [0 0], '-', 'Color', [.1 .1 .1 .5], 'linewidth', 1.25);

B1s_to_plot = [.6, .8, 1.0];

B1_indices_to_grab = [];

for ii = 1:numel(B1s_to_plot)
    [~, I] = min(abs(B1s_to_plot(ii) - B1s));
    B1_indices_to_grab(end + 1) = I;
end
ax = gca;

ax.Position(1) = ax.Position(1) + plot_horizontal_offset; % make room for label
ax.Position(3) = ax.Position(3) - plot_horizontal_offset; %

Zs_to_plot = Z_post_tipup(B1_indices_to_grab, :);
subplot(212);
plot(tipup_phases * 180/pi, Zs_to_plot', 'LineWidth', 2); hold on;
for ii = 1:numel(B1s_to_plot)
    legend_str{ii} = append('$\mathrm{B}_1 = ', num2str(B1s_to_plot(ii), '%.1f'), '$');
end
lgd = legend(legend_str, 'Location', 'south east', 'Interpreter','latex');
lgd.AutoUpdate = 'off';

ax = gca;
ax.XTick = 0:45:361;
ax.Position(1) = ax.Position(1) +plot_horizontal_offset; % make room for label
ax.Position(3) = ax.Position(3) - plot_horizontal_offset; %

xlabel('Tipup Phase [deg]', 'FontSize', 16);
ylabel({'Stored $\mathrm{Z}_0$', 'After Tipup'}, 'FontSize', 16);
grid on;


%new_yticks = ax.YTick;
%ax.YTick
if(tissue_type == 3)
    if(b_value == 0)
        new_yticks = [0];
        idx_to_label = [1, 2];
    else
        new_yticks = [0];
        idx_to_label = [1, 2, 3];
    end
else
    new_yticks = [min(ax.YTick) 0 max(ax.YTick)];
    %new_yticks = ax.YTick;
    idx_to_label = [1, 2];
end
colors = {[0 0.4470 0.7410], [0.8500 0.3250 0.0980], [0.9290 0.6940 0.1250]};
for idx = idx_to_label
    z_max_a =  round(max(Zs_to_plot(idx, :)), 3);
    z_min_a =  round(min(Zs_to_plot(idx, :)), 3);
    plot([0, 360], ones(2, 1) * z_max_a, ':', 'Color', colors{idx}, 'LineWidth', 1.5)
    plot([0, 360], ones(2, 1) * z_min_a, ':', 'Color', colors{idx}, 'LineWidth', 1.5)
    new_yticks(end+1) = z_max_a;
    new_yticks(end+1) = z_min_a;

%     z_mean = round(mean([z_min_a, z_max_a]), 3);
%     plot([0, 360], ones(2,1) * z_mean, ':', 'Color', colors{idx}, 'LineWidth', 1.5)
% 
%     if(~any(new_yticks == z_mean))
%         new_yticks(end + 1) = z_mean;
%     end
end
p = plot([0, 360], [0 0], '-', 'Color', [.1 .1 .1 .5], 'linewidth', 1.25);

%plot([0, 360], [0 0], '-', 'Color', [.6 .6 .6])
%text(363, 0, '0', 'FontSize', 14, 'HorizontalAlignment', 'left', 'VerticalAlignment', 'middle', 'Color', .25 * [.6 .6 .6])
ax.YTick = sort(new_yticks);

if(~skip_b_value_label)
    st = suptitle(parameters_tag);
    st.FontSize = 24;
    st.Position(1) = st.Position(1) + plot_horizontal_offset * .65;
end

ylim_max = max(abs(ax.YLim));
ylim([-ylim_max, ylim_max])
xlim([0, 360]);


if(letters.isKey(letter_key))
    ha = axes('pos',[0 1 1 1],'visible','off','Tag','letter');
    text(ha, .0 + plot_horizontal_offset/2, -.04, letters(letter_key), 'FontSize', 24, 'HorizontalAlignment', 'left'); % negative y to get in box

    if(do_left_label)
        text(ha, .0 + plot_horizontal_offset, -.5, {tissue_label, left_label}, ...
            'FontSize', 24, 'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'Rotation', 90, 'Interpreter', 'latex');
    end
end
%%
%export_fig(fig_b1_tissue_sims, sprintf('paper_figures/figures/b1_n_refoc_%d_tissue_%d_b_%d.png', N_refoc_pulses, tissue_type, b_value), '-m2.5')