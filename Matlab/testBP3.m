clc,clear

load('Data-Ass2.mat');
traindata = data(:,1:2000);
testdata = data(:,2001:3000);

insize = 2;%输入层神经元数目
hidesize = 10;%隐含层神经元数目
outsize = 2;%输出层神经元数目

yita1 = 0.001;%输入层到隐含层之间的学习率
yita2 = 0.001;%隐含层到输出层之间的学习率

W1 = rand(hidesize,insize);%输入层到隐含层之间的权重
W2 = rand(outsize,hidesize);%隐含层到输出层之间的权重
B1 = rand(hidesize,1);%隐含层神经元的阈值
B2 = rand(outsize,1);%输出层神经元的阈值

Y = zeros(2,2000);%期望输出
for i = 1:2000
    y = zeros(2,1);
    if traindata(3,i)==1
        y = [1;0];
    else
        y = [0;1];
    end
    Y(:,i) = y;
end

loop = 2000;
E = zeros(1,loop);
Acc = zeros(1,loop);
for loopi = 1:loop
    
    tag = 0;
    
    for i = 1:2000
        x = traindata(1:2,i);
        
        hidein = W1*x+B1;%隐含层输入值
        hideout = zeros(hidesize,1);%计算隐含层输出值
        for j = 1:hidesize
            hideout(j) = sigmod(hidein(j));
        end
        
        yin = W2*hideout+B2;%输入层输入值
        yout = zeros(outsize,1);%输出层输出值
        for j = 1:outsize
            yout(j) = sigmod(yin(j));
        end
        
        [~, M] = max(yout);
        [~, N] = max(Y(:,i));
        
        if M == N
            tag = tag + 1;
        end
        
        e = abs(yout-Y(:,i));%输出层计算结果误差
        E(loopi) = e(1)+e(2);
        %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        %后向反馈
        dB2 = zeros(outsize,1);%误差对输出层阈值求偏导，计算阈值变化量
        for j = 1:outsize
            dB2 = sigmod(yin(j))*(1-sigmod(yin(j)))*e(j)*yita2;
        end
        
        %隐含层与输出层之间的权重的变化量
        dW2 = zeros(outsize,hidesize);
        for j = 1:outsize
            for k = 1:hidesize
                dW2(j,k) = sigmod(yin(j))*(1-sigmod(yin(j)))*hideout(k)*e(j)*yita2;
            end
        end
        
        %隐含层阈值变化量
        dB1 = zeros(hidesize,1);
        for j = 1:hidesize
            tempsum = 0;
            for k = 1:outsize
                tempsum = tempsum + sigmod(yin(k))*(1-sigmod(yin(k)))*W2(k,j)*sigmod(hidein(j))*(1-sigmod(hidein(j)))*e(k)*yita1;
            end
            dB1(j) = tempsum;
        end
        
        %输入层到隐含层的权重变化量
        dW1 = zeros(hidesize,insize);
        for j = 1:hidesize
            for k = 1:insize
               tempsum = 0;
               for m = 1:outsize
                   tempsum = tempsum + sigmod(yin(m))*(1-sigmod(yin(m)))*W2(m,j)*sigmod(hidein(j))*(1-sigmod(hidein(j)))*x(k)*e(m)*yita1;
               end
               
               dW1(j,k) = tempsum;
                
            end
            
        end
        
        W1 = W1-dW1;
        W2 = W2-dW2;
        B1 = B1-dB1;
        B2 = B2-dB2;
        
    end
    
    Acc(loopi) = tag / 2000.0; 
    
    
    if mod(loopi,100)==0
        loopi
    end
    
end

plot(Acc);

% %查看训练效果
tempyout = zeros(2,1000);
for i = 1:1000
    x = testdata(1:2,i);
    
    hidein = W1*x+B1;%隐含层输入值
    hideout = zeros(hidesize,1);%隐含层输出值
    for j = 1:hidesize
        hideout(j) = sigmod(hidein(j));
    end
    
    yin = W2*hideout+B2;%输出层输入值
    yout = zeros(outsize,1);
    for j = 1:outsize
        yout(j) = sigmod(yin(j));
    end
    
    tempyout(:,i) = yout;
    
    if yout(1)>yout(2)
        scatter(x(1),x(2),'r')
        hold on;
    else
        scatter(x(1),x(2),'g')
        hold on;
    end
      
    
end