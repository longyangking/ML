nn = NeuralNetwork([2,3,1]);

X = [0,0;0,1;1,0;1,1];
y = [1,0,0,1];

nn = nn.fit(X,y,0.1,10000); % Need to check

testX = [0,0;0,1;1,0;1,1]; [M,N] = size(testX);
for x = 1:M
    fprintf('sample {%s} with value {%.2f}\n',num2str(testX(x,:)),nn.predict(testX(x,:)));
end

plot(log(nn.mse(1:10:end)));
xlabel('Error/dB');
ylabel('Epochs/Unit');