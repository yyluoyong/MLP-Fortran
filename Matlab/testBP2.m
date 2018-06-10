clc,clear  
Data = -3:0.01:3;  
xsize = size(Data);  
datasize = xsize(2);  
Value = zeros(1,datasize);  
for i = 1:datasize  
    Value(i) = sin(Data(i));  
end  
  
hidesize = 10;  
W1 = rand(hidesize,1);%�������������֮���Ȩ��  
B1 = rand(hidesize,1);%��������Ԫ����ֵ  
W2 = rand(1,hidesize);%�������������֮���Ȩ��  
B2 = rand(1,1);%�������Ԫ����ֵ  
yita = 0.05;  
  
loop = 50000;  
  
E = zeros(1,loop);%�������������ı仯  
Y = zeros(1,datasize);%ģ������Ľ��  
for loopi = 1:loop  
    tempsume = 0;  
    for i = 1:datasize  
        x = Data(i);%�������������  
        hidein = x*W1-B1;%���������������  
        hideout = zeros(hidesize,1);%��������������  
        for j = 1:hidesize  
            hideout(j) = sigmod(hidein(j));  
        end  
          
        y = W2*hideout-B2;%���  
          
        Y(i) = y;  
          
        e = y-Value(i);%���  
          
        %�������޸Ĳ���  
        dB2 = -1*yita*e;  
        dW2 = e*yita*hideout';  
        dB1 = zeros(hidesize,1);  
        for j = 1:hidesize  
            dB1(j) = W2(j)*sigmod(hidein(j))*(1-sigmod(hidein(j)))*(-1)*e*yita;  
        end  
          
        dW1 = zeros(hidesize,1);  
        for j = 1:hidesize  
            dW1(j) = W2(j)*sigmod(hidein(j))*(1-sigmod(hidein(j)))*x*e*yita;  
        end  
          
        W1 = W1-dW1;  
        B1 = B1-dB1;  
        W2 = W2-dW2;  
        B2 = B2-dB2;  
          
        tempsume = tempsume + abs(e);  
          
    end  
    E(loopi) = tempsume / datasize;  
      
    if mod(loopi,100)==0  
        fprintf('step = %d, error = %f\n', loopi, tempsume);  
    end  
end 

plot(Data, Y);
plot(Data, Y, '+', Data, sin(Data), ':');
