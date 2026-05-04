clear;
clc;
load('dataset_tratado.mat');

entry = matriz_norm'; 

catTarget = categorical(dados.class_cat);
indexTarget = double(catTarget);


targets = full(ind2vec(indexTarget'));

disp("Inputs e targets prontos!");

% --- Configurações de Rede ---
numNeuronios = 20; % Tentar mudar para 20, ou para [10 5] para 2 camadas ocultas
funcTreino = 'trainscg'; % Tentar: 'trainlm', 'trainbr', 'traingd', 'trainscg'
funcAtivacao = 'softmax'; % Tentar: 'purelin', 'tansig', 'logsig', 'softmax'
num_repeticoes = 10;

acertosTeste = zeros(1, num_repeticoes);
acertosGerais = zeros(1, num_repeticoes);

for i = 1:num_repeticoes
    net = patternnet(numNeuronios, funcTreino);
    
    net.layers{end}.transferFcn = funcAtivacao;
    
    net.divideParam.trainRatio = 70/100;
    net.divideParam.valRatio = 15/100;
    net.divideParam.testRatio = 15/100;
    net.trainParam.showWindow = false;
    
    [net, tr] = train(net, entry, targets);
    
    saida = net(entry);
    outIndices = vec2ind(saida);
    targetsIndices = vec2ind(targets);
    
    acertosGerais(i) = sum(outIndices == targetsIndices) / length(targetsIndices) * 100;
    
    % acertos exclusivos de teste
    saidaTeste = outIndices(tr.testInd);
    targetsTeste = targetsIndices(tr.testInd);
    acertosTeste(i) = sum(saidaTeste == targetsTeste) / length(targetsTeste) * 100;
    
    disp("Repetição " + num2str(i) + " concluída.");
end

disp("=================================================");
disp("Configuração: " + num2str(numNeuronios) + " neurónios | Função: " + funcTreino);
disp("-> Taxa de Acerto Geral (Média): " + num2str(mean(acertosGerais), '%.2f') + " %");
disp("-> Taxa de Acerto Teste (Média): " + num2str(mean(acertosTeste), '%.2f') + " %");
disp("=================================================");