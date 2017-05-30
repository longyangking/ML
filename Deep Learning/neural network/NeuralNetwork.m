classdef NeuralNetwork
    properties
        activation;
        activation_deriv;
        weights;
        layers;
        mse;
    end
    methods 
        function self =  NeuralNetwork(layers,activation,initialweights)
            addpath('activation');
            if nargin == 1
                self.activation = @tanh;
                self.activation_deriv = @tanh_deriv;
                initialweights = {};
            end
            if nargin == 2
                initialweights = {};
                % set Activation and Activation Derivation Function
                if strcmp(activation,'tanh')
                    self.activation = @tanh;
                    self.activation_deriv = @tanh_deriv;
                end
                if strcmp(activation,'logistic')
                    self.activation = @logistic;
                    self.activation_deriv = @logistic_deriv;
                end
                if strcmp(activation,'relu')
                    self.activation = @relu;
                    self.activation_deriv = @relu_deriv;
                end
            end

            self.layers = layers;

            if isempty(initialweights)
                self.weights = {};
                for layer = 2:(length(layers) - 1)
                    self.weights = [self.weights, (2*rand(layers(layer-1) + 1,layers(layer) + 1)-1)*0.5];
                end
                self.weights = [self.weights, (2*rand(layers(layer) + 1,layers(layer + 1))-1)*0.5];
            else
                self.weights = initialweights;
            end

            self.mse = [];
        end

        function self = fit(self,X,y,learning_rate,epochs)
            % Add Bias Term
            [M,N] = size(X); X = [X, ones(M,1)];

            % Stochastic Gradient Descent
            self.mse = [];            
            for k = 1:epochs
                index = randi(M);
                values = {X(index,:)};

                for layer = 1:length(self.weights)
                    values = [values, self.activation(values{layer}*self.weights{layer})];
                end
                errorvalue = y(index) - values{end};  self.mse = [self.mse,abs(errorvalue)^2];

                deltas = {errorvalue*self.activation_deriv(values{end})};
                for layer = (length(values)-1):-1:2
                    deltas = [deltas, (deltas{end}*(transpose(self.weights{layer}))).*self.activation_deriv(values{layer})];
                end
                deltas = deltas(end:-1:1);

                for layer = 1:length(self.weights)
                    value = values{layer};
                    delta = deltas{layer};
                    self.weights{layer} = self.weights{layer} + learning_rate*((transpose(value))*delta);
                end
            end
        end

        function preds = predict(self,X)
            % Add Bias Term
            [M,N] = size(X); X = [X, ones(M,1)];
            preds = ones(M,1);

            for index = 1:M
                pred = X(index,:);
                for layer = 1:length(self.weights)
                    pred = self.activation(pred*self.weights{layer});
                end
                preds(index) = pred;
            end
        end
    end
end