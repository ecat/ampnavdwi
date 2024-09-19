set(groot, 'defaulttextinterpreter', 'latex');
set(groot, 'defaultAxesTickLabelInterpreter', 'latex');
set(groot, 'defaultLegendInterpreter', 'latex');
set(groot, 'DefaultLineLineWidth', 1.);




get_matrix = @(w1, w2) [1 1; w1 w2];

N = 200;
w1s = linspace(-1, 1, N);
w2s = linspace(-1, 1, N);

reciprocal_condition_numbers = zeros(N);

for ii = 1:N
    for jj = 1:N
        M = get_matrix(w1s(ii), w2s(jj));
        reciprocal_condition_numbers(ii, jj) = 1 / cond(M);
    end
end

fig1= figure('Position', [100 100 650 650], 'Color', 'white'); 
ih = imagesc(reciprocal_condition_numbers, 'XData', w1s, 'YData', w2s);
xlabel('$w_1$', 'FontSize', 16); ylabel('$w_2$', 'FontSize', 16);
cbar = colorbar;
ylabel(cbar, 'Reciprocal of Condition Number', 'Interpreter', 'latex', 'FontSize', 14)
axis image;
caxis([0, 1.001])
h=gca; h.XAxis.TickLength = [0 0];
h.YAxis.TickLength = [0 0];
title({'Conditioning of No-Stabilizer Reconstruction with', 'Even k-space Lines Acquired Twice'}, ...
    'FontSize', 16)
set(gca, 'YDir', 'normal') % make it so that (-1, -1) is in bottom left, (1, 1) top right
ha = axes('pos',[0 1 1 1],'visible','off','Tag','letter');
text(ha, .01, -.1, 'A)', 'FontSize', 20, 'HorizontalAlignment', 'left'); % negative y to get in box

fig2 = figure('Position', [600 100 570 570], 'Color', 'white');
histogram(reciprocal_condition_numbers(:), 'Normalization', 'probability')
title({'PDF of Reciprocal Condition Numbers Assuming', ...
    '$w_1, w_2$ Independent and Uniformly Distributed'}, 'FontSize', 16)
xlabel('Reciprocal of Condition Number', 'FontSize', 14)
ylabel('Probability', 'FontSize', 14)
xlim([0, 1.])
ha = axes('pos',[0 1 1 1],'visible','off','Tag','letter');
text(ha, .01, -.06, 'B)', 'FontSize', 20, 'HorizontalAlignment', 'left'); % negative y to get in box

%%
%export_fig(fig1, 'figures/condition_number_a.png', '-m2.5')
%export_fig(fig2, 'figures/condition_number_b.png', '-m2.5')

