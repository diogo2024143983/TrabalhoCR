clear;
clc;

disp("a ler o dataset_tp");
dados=readtable("dataset_TP.csv");


%conversao
dados.maintenance_level= double (categorical(dados.maintenance_level, {'Low', 'Medium', 'High'}, 'Ordinal', true));


%modo de operaçao
dados.operating_mode = string(dados.operating_mode);
cateModo=unique(dados.operating_mode);
matrizModo=double (dados.operating_mode == cateModo');

%tipo de arerfecimento
dados.cooling_type = string(dados.cooling_type);
cateArrefecimento=unique(dados.cooling_type);
matrizArre=double(dados.cooling_type==cateArrefecimento');


%estado sensor
dados.sensor_status = string(dados.sensor_status);
cateSensor= unique(dados.sensor_status);
matrizSensor=double (dados.sensor_status==cateSensor');



dados = [dados, array2table(matrizModo), array2table(matrizArre), array2table(matrizSensor)];

dados.operating_mode = [];
dados.cooling_type = [];
dados.sensor_status = [];


%preencher os missing vlaues
colNum= {'temperature', 'vibration', 'rotation_speed', 'voltage','current', 'pressure', 'noise_level', 'efficiency','load_val', 'torque', 'maintenance_level'};

for i=1:length(colNum)
    nomeCol= colNum{i};
    if any(isnan(dados.(nomeCol)))
        dados.(nomeCol)(isnan(dados.(nomeCol))) = median(dados.(nomeCol), "omitnan");
    end
end


inputs_tabela = dados;
inputs_tabela.class_cat = []; 
matriz_entradas = table2array(inputs_tabela); 

% normalização min-max
minVal = min(matriz_entradas);
maxVal = max(matriz_entradas);
range_val = maxVal - minVal;
range_val(range_val == 0) = 1; %para evitar div por 0 (caso o min max sejam =)

matriz_norm = (matriz_entradas - minVal) ./ range_val;


%pesos para o cbr retrive

numAtributos = size(matriz_norm, 2);
pesos = ones(1, numAtributos);


idx_em_falta = find(ismissing(dados.class_cat));
idx_conhecidos = find(~ismissing(dados.class_cat));


for i = 1:length(idx_em_falta)
    linha_atual = idx_em_falta(i);
    
    diferencas = abs(matriz_norm(idx_conhecidos, :) - matriz_norm(linha_atual, :));
    distancias = sum(diferencas .* pesos, 2); 
    
    [~, idx_menor_dist] = min(distancias);
    
    dados.class_cat(linha_atual) = dados.class_cat(idx_conhecidos(idx_menor_dist));
end


save('dataset_tratado.mat', 'dados', 'matriz_norm', 'minVal', 'range_val', 'pesos', 'cateModo', 'cateArrefecimento', 'cateSensor');
disp('Tratamento de dados feito');