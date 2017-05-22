% Author: Yang Long
%
% E-mail: longyang_123@yeah.net

%% *********************************************************************************
%% To optimize the function 'x^2' and '(x-2)^2' simultaneously
fprintf('Optimizing Function 1...\n');
func = @(x)([x.^2,(x-2).^2]);
LB = [0]; UB = [2];
featuresize = 1; targetsize = 2;

psoopt = struct(...
    'Fitnessfunction', func, ...
    'Featuresize', featuresize, ...
    'Targetsize', targetsize, ...
    'LB', LB, ...
    'UB', UB, ...
    'IntCon',[],...     
    'Constraintfunction',@(x)(zeros(size(x,1),1)),...  % No constraint
    'ParticleSize',10,...
    'InitialParticles',[],...
    'InitialVelocities',[],...
    'C1',0.2,...
    'C2',0.2,...
    'w',1.0,...
    'MutationRate',0.2,...
    'MaxIteration',30,...
    'Verbose',false,... % Echo the process
    'Creationfunction',@psoCreation,...
    'Fitnessscalefunction',@psoFitnessScale,...
    'Distancefunction',@psoPareto,...
    'Mutationfunction',@psoMutation,...
    'Hyperfunction', @psoLog...    % To run after each itertion, log the PSO process
    );

[particles,objectives] = PSO(psoopt);

figure(1);
scatter(objectives(:,1),objectives(:,2));

%% *********************************************************************************
%% To optimize the function 'x1^2 + (x2-2)^2' and '|x1-2|+ x2^2' for two variables 'x1,x2' simultaneously
fprintf('Optimizing Function 2...\n');
func = @(x)([x(:,1).^2+(x(:,2)-2).^2,abs(x(:,1)-2)+(x(:,2)).^2]);
LB = [0,0]; UB = [2,2];
featuresize = 2; targetsize = 2;

psoopt = struct(...
    'Fitnessfunction', func, ...
    'Featuresize', featuresize, ...
    'Targetsize', targetsize, ...
    'LB', LB, ...
    'UB', UB, ...
    'IntCon',[2],...    
    'Constraintfunction',@(x)(zeros(size(x,1),1)),...  % No constraint
    'ParticleSize',20,...
    'InitialParticles',[],...
    'InitialVelocities',[],...
    'C1',0.2,...
    'C2',0.2,...
    'w',1.0,...
    'MutationRate',0.2,...
    'MaxIteration',20,...
    'Verbose',false,... % Echo the process
    'Creationfunction',@psoCreation,...
    'Fitnessscalefunction',@psoFitnessScale,...
    'Distancefunction',@psoPareto,...
    'Mutationfunction',@psoMutation,...
    'Hyperfunction',@(globalbestparticles,globalbestobjectives,psoopt,iter)(0) ...    % To run after each iteration, @psoLog
    );

[particles,objectives] = PSO(psoopt);

figure(2);
scatter(objectives(:,1),objectives(:,2));