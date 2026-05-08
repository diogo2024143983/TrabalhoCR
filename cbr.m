clear; 
clc;

disp("Raciocinio baseado em casos");

load('dataset_tratado.mat');
disp("Sucesso a carregar is dados");

dadosTeste=readtable('dataset_TP_test.csv');
disp ("Sucesso a carregar os dados de teste");

%preparar os dados do teste
dadosTeste.maintenance_level=cellstr(dadosTeste.maintenance_level);
dadosTeste.maintenance_level=double(categorical(dadosTeste.maintenance_level, {'Low', 'Medium', 'High'}, 'Ordinal', true));

dadosTeste.operating_mode = string(dadosTeste.operating_mode);
matrizModoTeste = double(dadosTeste.operating_mode == cateModo');

dadosTeste.cooling_type = string(dadosTeste.cooling_type);
matrizArreTeste = double(dadosTeste.cooling_type == cateArrefecimento');

dadosTeste.sensor_status = string(dadosTeste.sensor_status);
matrizSensorTeste = double(dadosTeste.sensor_status == cateSensor');


dadosTeste = [dadosTeste, array2table(matrizModoTeste), array2table(matrizArreTeste), array2table(matrizSensorTeste)];
dadosTeste.operating_mode = [];
dadosTeste.cooling_type = [];
dadosTeste.sensor_status = [];

% guardar as respostas certas para calcular a % de acerto da classificacao
classesReais = dadosTeste.class_cat;

inputsTeste=dadosTeste;
inputsTeste.class_cat=[];
matrizTesteEntradas=table2array(inputsTeste);

matNormTeste=(matrizTesteEntradas-minVal)./range_val;

% matrizes nao normalizadas para a comparacao
dadosInputs = dados;
dadosInputs.class_cat = [];
matrizNaoNorm = table2array(dadosInputs);

disp("dados do teste preparadoscom sucesso");


% Testar a classificação 

disp("");
disp("<<<<<<<<<<<<<< Testes de classificacao >>>>>>>>>>>>>>>");

pesos_iguais = ones(1, size(matriz_norm, 2));

pesos_sensores = ones(1, size(matriz_norm, 2));
pesos_sensores(2:4) = 5; % dar mais peso a vibration, rotation e voltage

configs = {
    'Normalizado + Pesos Iguais', matriz_norm, matNormTeste, pesos_iguais;
    'Normalizado + Pesos Sensores (2,3,4)', matriz_norm, matNormTeste, pesos_sensores;
    'Nao Normalizado + Pesos Iguais', matrizNaoNorm, matrizTesteEntradas, pesos_iguais
};

for c = 1:size(configs, 1)
    nomeConf = configs{c, 1};
    base = configs{c, 2};
    testes = configs{c, 3};
    pesos_atuais = configs{c, 4};
    
    previsoes = cell(height(dadosTeste), 1);
    
    for i = 1:height(dadosTeste)
        novo = testes(i, :);
        
        difs = abs(base - novo);
        dists = sum(difs .* pesos_atuais, 2);
        
        [~, idxMenor] = min(dists);
        previsoes{i} = dados.class_cat{idxMenor};
    end
    
    acertos = sum(strcmp(previsoes, classesReais));
    taxa = (acertos / height(dadosTeste)) * 100;
    
    disp("-> " + nomeConf + " | Taxa de acerto: " + num2str(taxa) + "%");
end
disp("----------------------------------------------");
disp("");



%parte dos 4 R

%retrive

Lsemelhanca= 0.85;

casosRec=cell(height(dadosTeste),1);

for i=1:height(dadosTeste)
    disp("a ver caso n "+num2str(i));

    casoNovo=matNormTeste(i, :);

    %diferenca entre o novo e os da base de dados
    dif=abs(matriz_norm - casoNovo);
    distancias= sum(dif .*pesos, 2);


    semelhancas= 1-(distancias/sum(pesos));


    indexSimi=find(semelhancas >= Lsemelhanca);

    if isempty(indexSimi)
        disp("Nenhum caso com mais de "+ num2str(Lsemelhanca));
    else
        disp("similares encontrados: "+ num2str(length(indexSimi)));

        [maiorSemelhanca, melhorIndex]=max(semelhancas);
        disp("com mais parecencas: "+ num2str(melhorIndex)+"do ficheiro com "+num2str(maiorSemelhanca*100)+ "%");
    end

    casosRec{i}=indexSimi;
    disp(".-.-.-.-.-.-.-.-.-.-")

end


% reutilizar
disp("---------parte 2------------")

load ('redeTreinada.mat', 'net');


tempSugeridas=zeros(height(dadosTeste), 1);

for i=1:height(dadosTeste)

    vibTeste=dadosTeste.vibration(i);
    velTeste=dadosTeste.rotation_speed(i);
    tensaoTeste=dadosTeste.voltage(i);

    testesEntry=[vibTeste; velTeste;tensaoTeste];

    tempPrevista=net(testesEntry);

    %temos que guradar para a revise
    tempSugeridas(i) = tempPrevista;

disp("Caso n " + num2str(i) + ": A Rede sugere Temperatura = " + num2str(tempPrevista) + " graus.");
end



disp("");

% rever
disp("----------parte 3-----------");

erros=zeros(height(dadosTeste), 1);

for i=1:height(dadosTeste)

    tempPrevista=tempSugeridas(i);

    tempReal=dadosTeste.temperature(i);

    erroAbsoluto= abs(tempPrevista-tempReal);
    erros(i)= erroAbsoluto;

disp("Caso n " + num2str(i) + " | Previsto: " + num2str(tempPrevista) + " | Real: " + num2str(tempReal) + " | Erro: " + num2str(erroAbsoluto));  

altera= input("Quer alterar a temperatura real para a da do valor previsto? (s/n)", 's');

if altera=="s"
    dadosTeste.temperature(i)=tempPrevista;
    disp("temperatura alterada");
    disp("-> Temperatura atualizada de " + num2str(tempReal) + " para " + num2str(dadosTeste.temperature(i)));
else
    disp("temperatura mantida");

end
disp("-<-<-<-<-<>->->->-")
end

erroMedio= mean(erros);
disp("<<<<<<<<<<<<<<  Erro médio: " + num2str(erroMedio)+" >>>>>>>>>>>>>>>>>>>");


% reter
disp("<<<<<<<<<<<<<< reter >>>>>>>>>>>>>>>");

limiteERRO = 4.0;
casosRetidos = 0;

for i=1:height(dadosTeste)
    if erros(i) <= limiteERRO
        casosRetidos = casosRetidos + 1;
        
        novoCasoDados = dadosTeste(i, :);
        novoCasoDados.Properties.VariableNames = dados.Properties.VariableNames;
        dados = [dados; novoCasoDados];
        matriz_norm = [matriz_norm; matNormTeste(i, :)];
    end
end

save('dataset_tratado.mat', 'dados', 'matriz_norm', '-append');

disp("Total de casos novos avaliados: " + num2str(height(dadosTeste)));
disp("Casos retidos na memória (Erro <= " + num2str(limiteERRO) + " graus): " + num2str(casosRetidos));