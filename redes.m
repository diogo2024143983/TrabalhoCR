clear;
clc;

disp("carregar historico");
load('dataset_tratado.mat', 'dados');

inputs= [dados.vibration, dados.rotation_speed, dados.voltage]';

prever=dados.temperature';

disp("treino");

net=fitnet(10);
%net.trainParam.showWindow=false;
net=train(net,inputs,prever);

save('redeTreinada.mat', 'net');
disp("treino feito");