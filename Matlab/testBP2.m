clc,clear  
Data = -3:0.01:3;  
xsize = size(Data);  
datasize = xsize(2);  
Value = zeros(1,datasize);  
for i = 1:datasize  
    Value(i) = sin(Data(i));  
end  
  
hidesize = 10;  
W1 = rand(hidesize,1);%输入层与隐含层之间的权重  
B1 = rand(hidesize,1);%隐含层神经元的阈值  
W2 = rand(1,hidesize);%隐含层与输出层之间的权重  
B2 = rand(1,1);%输入层神经元的阈值  
yita = 0.05;  
  
loop = 50000;  
  
E = zeros(1,loop);%误差随迭代次数的变化  
Y = zeros(1,datasize);%模型输出的结果  
for loopi = 1:loop  
    tempsume = 0;  
    for i = 1:datasize  
        x = Data(i);%输入层输入数据  
        hidein = x*W1-B1;%隐含层的输入数据  
        hideout = zeros(hidesize,1);%隐含层的输出数据  
        for j = 1:hidesize  
            hideout(j) = sigmod(hidein(j));  
        end  
          
        y = W2*hideout-B2;%输出  
          
        Y(i) = y;  
          
        e = y-Value(i);%误差  
          
        %反馈，修改参数  
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
