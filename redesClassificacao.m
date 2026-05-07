clear;
clc;
load('dataset_tratado.mat');

inputs = matriz_norm';

catTarget = categorical(dados.class_cat);
indexTarget = double(catTarget);
targets = full(ind2vec(indexTarget'));

configs = {
    'Defeito (10, trainlm)', patternnet(10);
    'Conf1 (5-5, trainlm)', patternnet([5 5]);
    'Conf2 (10-10, trainlm)', patternnet([10 10]);
    'Conf3 (20, trainscg)', patternnet(20);
    'Conf4 (10, traingd)', patternnet(10);
    'Conf5 (10, trainbr)', patternnet(10);
    'Conf6 (10, trainscg)', patternnet(10);
    'Conf7 (10, logsig)', patternnet(10);
    'Conf8 (10, tansig)', patternnet(10);
    'Conf9 (10, softmax)', patternnet(10);
    'Conf10 (60/20/20)', patternnet(10);
    'Conf11 (80/10/10)', patternnet(10)
};

configs{4,2}.trainFcn = 'trainscg';
configs{5,2}.trainFcn = 'traingd';
configs{6,2}.trainFcn = 'trainbr';
configs{7,2}.trainFcn = 'trainscg';

configs{8,2}.layers{2}.transferFcn = 'logsig';
configs{9,2}.layers{2}.transferFcn = 'tansig';
configs{10,2}.layers{2}.transferFcn = 'softmax';

configs{11,2}.divideParam.trainRatio = 0.60;
configs{11,2}.divideParam.valRatio = 0.20;
configs{11,2}.divideParam.testRatio = 0.20;

configs{12,2}.divideParam.trainRatio = 0.80;
configs{12,2}.divideParam.valRatio = 0.10;
configs{12,2}.divideParam.testRatio = 0.10;

resultados = zeros(12, 2);
melhor_acc = 0; pior_acc = 100;

for i = 1:12
    net_atual = configs{i, 2};
    nome = configs{i, 1};
    net_atual.trainParam.showWindow = false;
    
    acc_global = zeros(1, 10);
    acc_teste = zeros(1, 10);
    
    for rep = 1:10
        [net_treinada, tr] = train(net_atual, inputs, targets);
        
        prev = net_treinada(inputs);
        c_prev = vec2ind(prev);
        c_real = vec2ind(targets);
        acc_global(rep) = sum(c_prev == c_real) / length(c_real);
        
        prev_t = prev(:, tr.testInd);
        targ_t = targets(:, tr.testInd);
        acc_teste(rep) = sum(vec2ind(prev_t) == vec2ind(targ_t)) / length(tr.testInd);
    end
    
    resultados(i, 1) = mean(acc_global) * 100;
    resultados(i, 2) = mean(acc_teste) * 100;
    
    if resultados(i, 1) > melhor_acc
        melhor_acc = resultados(i, 1);
        melhor_targ = targ_t; melhor_prev = prev_t; melhor_nome = nome;
    end
    if resultados(i, 1) < pior_acc
        pior_acc = resultados(i, 1);
        pior_targ = targ_t; pior_prev = prev_t; pior_nome = nome;
    end
    
    fprintf('%s | Global: %.2f%% | Teste: %.2f%%\n', nome, resultados(i,1), resultados(i,2));
end

figure('Name', 'Melhor Rede');
plotconfusion(melhor_targ, melhor_prev);
title(['Melhor: ' melhor_nome]);

figure('Name', 'Pior Rede');
plotconfusion(pior_targ, pior_prev);
title(['Pior: ' pior_nome]);