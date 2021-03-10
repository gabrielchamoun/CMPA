clc; close all; clear all;
set(0, 'DefaultFigureWindowStyle', 'docked')
warning('off','all')




% Part 1: Generating diode data
% Paramters
Is = 0.01e-12;
Ib = 0.1e-12;
Gp = 0.1;
Vb = 1.3;
V = linspace(-1.95, 0.7, 200);
% Diode Equation
I = Is.*(exp(V.*1.2/0.025)-1) ...
    + Gp.*V ...
    - Ib.*(exp(-(V+Vb).*1.2/0.025)-1);
% random variation to represent experimental noise
I_rnd = I + I.*0.2.*rand(1,200);




% Part 2: Fitting a 4 and 8 order polynomial to current
I_poly4 = polyval(polyfit(V,I,4), V);
I_poly8 = polyval(polyfit(V,I,8), V);
% random variation fit
I_rnd_poly4 = polyval(polyfit(V,I_rnd,4), V);
I_rnd_poly8 = polyval(polyfit(V,I_rnd,8), V);

subplot(3,2,1); hold on; 
plot(V,I); plot(V,I_poly4); plot(V,I_poly8);
title('Polynomial fitting to diode equation');
xlabel('V','FontSize',12), ylabel('Id (A)','FontSize',12);
legend('Id', 'poly4', 'poly8');

subplot(3,2,2); hold on;
title('Polynomial fitting to diode equation with variance');
plot(V,I_rnd); plot(V,I_rnd_poly4); plot(V,I_rnd_poly8);
xlabel('V','FontSize',12), ylabel('Id (A)','FontSize',12);
legend('Id', 'poly4', 'poly8');




% Part 3: Nonlinear curve fitting
fo = fittype('A.*(exp(1.2*x/25e-3)-1)+B.*x-C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff = fit(V', I', fo);
If4 = ff(V);

fo = fittype('A.*(exp(1.2*x/25e-3)-1)+B.*x-C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff = fit(V', I', fo);
If3 = ff(V);

fo = fittype('A.*(exp(1.2*x/25e-3)-1)+0.1.*x-C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff = fit(V', I', fo);
If2 = ff(V);

% Nonlinear fitting to the random variance
fo = fittype('A.*(exp(1.2*x/25e-3)-1)+B.*x-C*(exp(1.2*(-(x+D))/25e-3)-1)');
ff = fit(V', I_rnd', fo);
If4_rnd = ff(V);

fo = fittype('A.*(exp(1.2*x/25e-3)-1)+B.*x-C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff = fit(V', I_rnd', fo);
If3_rnd = ff(V);

fo = fittype('A.*(exp(1.2*x/25e-3)-1)+0.1.*x-C*(exp(1.2*(-(x+1.3))/25e-3)-1)');
ff = fit(V', I_rnd', fo);
If2_rnd = ff(V);

subplot(3,2,3); hold on; 
plot(V,I); plot(V, If2); plot(V, If3); plot(V, If4);
title('Nonlinear fitting to diode equation');
xlabel('V','FontSize',12), ylabel('Id (A)','FontSize',12);
legend('Id', '2 of 4 fit', '3 of 4 fit', '4 of 4 fit');
axis([-2 1 -5 5]);

subplot(3,2,4); hold on; 
plot(V,I_rnd); plot(V, If2_rnd); plot(V, If3_rnd); plot(V, If4_rnd);
title('Nonlinear fitting to diode equation with variance');
xlabel('V','FontSize',12), ylabel('Id (A)','FontSize',12);
legend('Id', '2 of 4 fit', '3 of 4 fit', '4 of 4 fit');
axis([-2 1 -5 5]);




% Part 4: Fitting using the Neural Net
inputs = V;
targets = I;
hiddenLayerSize = 10;
net = fitnet(hiddenLayerSize);
net.divideParam.trainRatio = 70/100;
net.divideParam.valRatio = 15/100;
net.divideParam.testRatio = 15/100;
[net,tr] = train(net,inputs,targets);
Inn = net(inputs);
% errors = gsubtract(outputs,targets);
% performance = perform(net,targets,Inn);

% Again for variation of I
targets = I_rnd;
[net,tr] = train(net,inputs,targets);
Inn_rnd = net(inputs);

subplot(3,2,5); hold on; 
plot(V,I); plot(V, Inn);
title('Neural Network fitting to diode equation');
xlabel('V','FontSize',12), ylabel('Id (A)','FontSize',12);
legend('Id', 'Neural Solve');
axis([-2 1 -5 5]);

subplot(3,2,6); hold on; 
plot(V,I_rnd); plot(V, Inn_rnd);
title('Neural Network fitting to diode equation with variance');
xlabel('V','FontSize',12), ylabel('Id (A)','FontSize',12);
legend('Id', 'Neural Solve');
axis([-2 1 -5 5]);
hold off