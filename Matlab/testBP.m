clc;
clear;

P = [-1:0.05:1];

T = sin(2*pi*P);

figure

plot(P,T,'+');

hold on;

plot(P, sin(2*pi*P), ':');

%net = newff(minmax(P), [20, 1], {'tansig', 'purelin'});
net = newff(P, T, 6);

net.trainFcn = 'trainbr';

net.trainParam.show = 50;
net.trainParam.lr = 0.05;
net.trainParam.epochs = 500;
net.trainParam.goal = 1e-3;

[net, tr] = train(net, P, T);

A = sim(net, P);

E = T - A;

MSE = mse(E);

plot(P, A, P, T, '+', P, sin(2*pi*P), ':');

legend('Ñù±¾µã', 'sin', 'sim');