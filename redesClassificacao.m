clear;
clc;
load('dataset_tratado.mat');
inputs = matriz_norm';
catTarget = categorical(dados.class_cat);
indexTarget = double(catTarget);
targets = full(ind2vec(indexTarget'));

% para podermos testar a alinea dos dados nao normalizados
dados_sem_target = dados;
dados_sem_target.class_cat = [];
inputs_nao_norm = table2array(dados_sem_target)';

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
redes_guardadas = cell(12, 1); % precisamos disto para não apagar as redes depois do ciclo

disp("a treinar as configurações todas...");

for i = 1:12
    net_atual = configs{i, 2};
    nome = configs{i, 1};
    net_atual.trainParam.showWindow = false;
    
    acc_global = zeros(1, 10);
    acc_teste = zeros(1, 10);
    melhor_rep_acc = 0; % ajuda a guardar a melhor das 10 repeticoes
    
    for rep = 1:10
        [net_treinada, tr] = train(net_atual, inputs, targets);
        
        prev = net_treinada(inputs);
        c_prev = vec2ind(prev);
        c_real = vec2ind(targets);
        acc_global(rep) = sum(c_prev == c_real) / length(c_real);
        
        prev_t = prev(:, tr.testInd);
        targ_t = targets(:, tr.testInd);
        acc_teste(rep) = sum(vec2ind(prev_t) == vec2ind(targ_t)) / length(tr.testInd);
        
        if acc_global(rep) > melhor_rep_acc
            melhor_rep_acc = acc_global(rep);
            redes_guardadas{i} = net_treinada;
        end
    end
    
    resultados(i, 1) = mean(acc_global) * 100;
    resultados(i, 2) = mean(acc_teste) * 100;
    
    fprintf('%s | Global: %.2f%% | Teste: %.2f%%\n', nome, resultados(i,1), resultados(i,2));
end

disp("---------------------------------");

% em vez de usar os if como tinhas, usamos sort para ordenar logo as 12
[~, idx_ordem] = sort(resultados(:,1), 'descend');
top3_idx = idx_ordem(1:3);
piores3_idx = idx_ordem(end-2:end);

disp("As 3 melhores:");
for k = 1:3
    disp(num2str(k) + "- " + configs{top3_idx(k), 1} + " com " + num2str(resultados(top3_idx(k), 1)) + "%");
end

disp("As 3 piores:");
for k = 1:3
    disp(num2str(k) + "- " + configs{piores3_idx(k), 1} + " com " + num2str(resultados(piores3_idx(k), 1)) + "%");
end

% guardar as 3 redes 
rede1 = redes_guardadas{top3_idx(1)};
rede2 = redes_guardadas{top3_idx(2)};
rede3 = redes_guardadas{top3_idx(3)};
save('melhores_redes.mat', 'rede1', 'rede2', 'rede3');
disp("-> melhores redes gravadas no melhores_redes.mat");

disp("---------------------------------");
disp("A comparar dados Normalizados vs Nao Normalizados:");

indices_comparar = [top3_idx; piores3_idx];

for k = 1:length(indices_comparar)
    idx = indices_comparar(k);
    net_limpa = configs{idx, 2};
    net_limpa.trainParam.showWindow = false;

    net_nn = train(net_limpa, inputs_nao_norm, targets);
    prev_nn = net_nn(inputs_nao_norm);
    acc_nn = sum(vec2ind(prev_nn) == vec2ind(targets)) / length(targets) * 100;

    disp(configs{idx, 1} + " -> Norm: " + num2str(resultados(idx,1)) + "% | Nao Norm: " + num2str(acc_nn) + "%");
end

disp("---------------------------------");
disp("A testar as melhores redes com o dataset_TP_test.csv...");

dadosTeste = readtable('dataset_TP_test.csv');
dadosTeste.maintenance_level = double(categorical(cellstr(dadosTeste.maintenance_level), {'Low', 'Medium', 'High'}, 'Ordinal', true));
matrizModoTeste = double(string(dadosTeste.operating_mode) == cateModo');
matrizArreTeste = double(string(dadosTeste.cooling_type) == cateArrefecimento');
matrizSensorTeste = double(string(dadosTeste.sensor_status) == cateSensor');

dadosTeste = [dadosTeste, array2table(matrizModoTeste), array2table(matrizArreTeste), array2table(matrizSensorTeste)];
dadosTeste.operating_mode = [];
dadosTeste.cooling_type = [];
dadosTeste.sensor_status = [];

targetTeste_str = categorical(dadosTeste.class_cat);
targets_teste = full(ind2vec(double(targetTeste_str)'));

dadosTeste.class_cat = [];
matrizTesteEntradas = table2array(dadosTeste);

matNormTeste = (matrizTesteEntradas - minVal) ./ range_val;
inputs_teste = matNormTeste';

redes_top = {rede1, rede2, rede3};
for k = 1:3
    net_teste = redes_top{k};
    prev_teste = net_teste(inputs_teste);

    acc_t = sum(vec2ind(prev_teste) == vec2ind(targets_teste)) / length(targets_teste) * 100;
    disp("Rede Top " + num2str(k) + " (" + configs{top3_idx(k), 1} + ") acertou " + num2str(acc_t) + "%");

    figure('Name', ['Top ' num2str(k) ' - ' configs{top3_idx(k), 1}]);
    plotconfusion(targets_teste, prev_teste);
    title(['Teste Top ' num2str(k) ' - ' configs{top3_idx(k), 1}]);
end