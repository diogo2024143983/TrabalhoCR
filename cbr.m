
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



inputsTeste=dadosTeste;
inputsTeste.class_cat=[];
matrizTesteEntradas=table2array(inputsTeste);

matNormTeste=(matrizTesteEntradas-minVal)./range_val;

disp("dados do teste preparadoscom sucesso");

%----------------------------------------------

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
end

erroMedio= mean(erros);

disp("<<<<<<<<<<<<<<  Erro médio: " + num2str(erroMedio)+" >>>>>>>>>>>>>>>>>>>");


% reter
disp("<<<<<<<<<<<<<< reter >>>>>>>>>>>>>>>");

limiteERRO=4.0;
casosRetidos=0;

for i=1:height(dadosTeste)
    if erros(i)<= limiteERRO
        casosRetidos = casosRetidos + 1;
    end
end
disp("Total de casos novos avaliados: " + num2str(height(dadosTeste)));
disp("Casos retidos na memória (Erro <= " + num2str(limiteERRO) + " graus): " + num2str(casosRetidos));