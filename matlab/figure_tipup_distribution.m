set(groot, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'DefaultLineLineWidth', 1.);

%%
N_measurements = [4, 8, 12, 24];
N = numel(N_measurements);

mean_offset = 1;

if(mean_offset == 0)
    theta_mean = 0;
    mean_labels = {'0'};    
else
    theta_mean = pi/3;
    mean_labels = {'\pi/3'};
end


%theta_distribution = 'Uniform'
theta_distribution = 'Gaussian'
if strcmp(theta_distribution, 'Uniform')
    theta_max = [1.5 * pi, pi, pi/2];
    theta_labels = {'[$-3\pi/4, 3\pi/4$))', '[$-\pi/2, \pi/2$)', '[$-\pi/4, \pi/4$)'};

    distribution_labels = {};
    for jj = 1:3
        distribution_labels{jj} = strcat('$\Theta \sim \mathrm{Uniform}($', theta_labels{jj}, '$)$');
    end
else
    theta_max = [.5 * pi, .33 * pi, .25 * pi];
    theta_labels = {'$\pi/2$', '$\pi/3$', '$\pi/4$'};
    distribution_labels = {};
    for jj = 1:3
        distribution_labels{jj} = strcat('$\Theta \sim \mathcal{N}(', mean_labels{1}, ', (', strip(theta_labels{jj}, 'both', '$') , ')^2)$');
    end
end

M = numel(theta_max);

N_montecarlo = 1e5;

S_reconstructed = zeros(N, M, N_montecarlo);
N_acceptable_measurements = zeros(N, M, N_montecarlo);
acceptable_error = .1;


for ii = 1:N
    N_meas = N_measurements(ii);
    
    for jj = 1:M
        if(strcmp(theta_distribution, 'Uniform'))
            theta = rand(N_meas, N_montecarlo) * theta_max(jj) - theta_max(jj) / 2; % center on zero
        else
            theta = randn(N_meas, N_montecarlo) * theta_max(jj);
            std(theta(:))
        end
        
        theta = theta + theta_mean;
        S = cos(theta);
        
        N_acceptable_measurements(ii, jj, :) = sum(abs(S) > (1 - acceptable_error), 1);
        
        S_r = max(abs(S), [], 1);
        S_reconstructed(ii, jj, :) = S_r;
    end    
   
end
   
%max(N_acceptable_measurements, [], 3)
mean_reconstructed_signal = mean(S_reconstructed, 3)

%% convert to cdfs
bin_width = .005;
bins = 0:bin_width:1;
cdf_Sr = zeros(N, M, numel(bins));

for ii = 1:N
    for jj = 1:M
        for bb = 1:numel(bins)
            cdf_Sr(ii, jj, bb) = numel(find(S_reconstructed(ii, jj, :) < bins(bb))) / N_montecarlo;
        end
    end
end

pdf_Sr = diff(cdf_Sr, [], 3);

cdf_N_acceptable_measurements = zeros(N, M, max(N_measurements) + 1);

for ii = 1:N
    for jj = 1:M        
        N_tmp = N_measurements(ii);
        for bb = 0:N_tmp
            % inclusive here so when subtract from 1 it is exclusive
            cdf_N_acceptable_measurements(ii, jj, bb + 1) = numel(find(N_acceptable_measurements(ii, jj, :) <= bb)) / N_montecarlo;
            
        end
    end
end

%%
cmap = {[0 0.4470 0.7410],[0.8500 0.3250 0.0980],[0.9290 0.6940 0.1250], [0.4940 0.1840 0.5560]};
% cmap = brewermap(4, 'Set1');

fig1 = figure('Position', [100 100 550 800], 'Color', 'white');

for jj = 1:M
    subplot(M, 1, jj);
    for ii = 1:N
        %cdfplot(S_reconstructed(ii, jj, :))
        p = plot(bins, squeeze(1 - cdf_Sr(ii, jj, :)), 'LineWidth', 1.5);
        hold on;
    end

    plot([.9, .9], [0, 1], '--k', 'LineWidth', 1.5)
    plot([0, 1], [.9, .9], ':k', 'LineWidth', 1.)

    for ii = 1:N
        plot(ones(2, 1) * mean_reconstructed_signal(ii, jj), [0 1], ':', 'LineWidth', 1.5, 'Color', cmap{ii});
    end

    rectangle('Position', [.9, .9, .1, .1], 'FaceColor', [.2 .2 .2 .1], 'EdgeColor', 'none')
    if jj == 2
        ylbl = ylabel('$\mathrm{P}(S_r > s) = 1 - \mathrm{CDF}(s)$', 'Rotation', 90, 'FontSize', 14);
    end
    
    title(distribution_labels{jj}, 'Interpreter', 'latex', 'FontSize', 14)

    do_draw_mean_label = 1;
    
    if(do_draw_mean_label && jj == 1)
        for ii = 1:N
            mean_text = sprintf('%.3f', round(mean_reconstructed_signal(ii, jj), 3));
            text(mean_reconstructed_signal(ii, jj) + .004, -.02, mean_text, 'Color', cmap{ii}, 'Rotation', -90, 'VerticalAlignment', 'top');
        end
        
    end
    
    ax = gca;
    ax.XTick = [.75:.05:1];
    
    grid on;
    
    xlim([0.75, 1]);
    ylim([0, 1.01])
    
end
xlabel('Signal $s$', 'FontSize', 14, 'Interpreter', 'latex')
    
legend_strs = {};
for ii = 1:N
    legend_strs{ii} = sprintf('%d Excitations', N_measurements(ii));
end
subplot(M, 1, 2);
legend(legend_strs, 'Interpreter', 'latex', 'Location', 'south west', 'FontSize', 12);
st = suptitle({'Probability of Reconstructed Signal Being Greater Than $s$', 'for Different $\Theta$ and Number of Excitations'});
st.FontSize = 16;
st.Position(2) = st.Position(2) + .015;


ha = axes('pos',[0 1 1 1],'visible','off','Tag','letter');
text(ha, .01, -.1, 'A)', 'FontSize', 24, 'HorizontalAlignment', 'left'); % negative y to get in box

%%
fig2 = figure('Position', [800 100 700 800], 'Color', 'white');

XTicks = {0:4, 0:2:8, 0:2:16, 0:4:24};

for ii = 1:N
    subplot(N, 1, ii);
    for jj = 1:M
        %h = histogram(N_acceptable_measurements(ii, jj, :), 'BinWidth', 1, 'Normalization', 'cdf',...
        %    'FaceAlpha', .1, 'EdgEcolor', cmap(jj, :), 'FaceColor', cmap(jj, :));  
        N_tmp = N_measurements(ii);
        p = plot(0:N_tmp, 1 - squeeze(cdf_N_acceptable_measurements(ii, jj, 1:N_tmp + 1)), ':o', 'LineWidth', 1.5);
        ax = gca();
        ax.XTick = XTicks{ii};
        hold on;
    end
    grid on;
    title(sprintf('%d Excitations', N_measurements(ii)), 'FontSize', 14);
    xlim([0, N_measurements(ii)]);
    ylim([0,1.02])
    if ii == 2
        ylbl = ylabel('$\mathrm{P}(N > n) = 1 - \mathrm{CDF}(n)$', 'Rotation', 90, 'FontSize', 14);
        ylbl.Position(2) = ylbl.Position(2) - .75;
        ylbl.Position(1) = ylbl.Position(1) - .1;
    end
    
    if ii == 1
        legend(distribution_labels, 'Location', 'north east', 'FontSize', 12);
    end
end
xlabel('Number of Measurements $n$ Within 10\% of Maximum Signal', 'FontSize', 14, 'Interpreter', 'latex')
st = suptitle({'Probability of Having More Than $n$ Measurements Within 10\%', 'of Maximum Signal for Different $\Theta$ and Number of Excitations'});
st.FontSize = 16;
st.Position(2) = st.Position(2) + .015;

ha = axes('pos',[0 1 1 1],'visible','off','Tag','letter');
text(ha, .03, -.1, 'B)', 'FontSize', 24, 'HorizontalAlignment', 'left'); % negative y to get in box
%%

%export_fig(fig1, sprintf('figures/montecarlo_pdfa_thetamean_%.2f.png', theta_mean), '-m2.5');
%export_fig(fig2, sprintf('figures/montecarlo_pdfb_thetamean_%.2f.png', theta_mean), '-m2.5');
    