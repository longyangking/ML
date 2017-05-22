% Author: Yang Long
%
% E-mail: longyang_123@yeah.net

psoopt = struct(...
    'Fitnessfunction', %% The function to calculate the fitness
    'Featuresize', %% The number of features
    'Targetsize', %% The number of optimizing target
    'LB',[], ...   %% Lower boundary for variables
    'UB',[], ...    %% Upper boundary for variabels
    'IntCon',[],...   %% Integer Constraint
    'Constraintfunction',@(x)0,...  % Constraint function setting
    'ParticleSize',100,...  %% Particles size in PSO
    'InitialParticles',[],...   %% Set initial particles values
    'InitialVelocities',[],...  %% Set initial velocities values
    'C1',0.2,...    %% Factor C1
    'C2',0.2,...    %% Factor C2
    'w',1.0,...     %% Inertia weight of velocity
    'MutationRate',0.2,...  %% Rate of mutation
    'MaxIteration',100,...  %% Maximum iteration in PSO
    'Verbose',false,...     %% To print the info in terminal
    'Creationfunction',@psoCreation,... %% Set function to create initial particle swarm
    'Fitnessscalefunction',@psoFitnessScale,... %% Set function to rescale the fitness values
    'Distancefunction',@psoPareto,...   %% Set function to calculate the Pareto front position of each particle
    'Mutationfunction',@psoMutation,... %% Set function to mutate the particles
    'Hyperfunction',@psoLog     % To run after each iteration, psoLog is the function to write a log during the process of PSO
    );

%% To optimize with PSO, just send this 'psoopt' structure to the function 'PSO'.
%% After completion of calculation, the function 'PSO' will return the particles and 
%% objectives in Pareto forefront.
%% Usage:
%%          [particles,objectives] = PSO(psoopt);
%%
%% You can draw the result by the following code example:
%%
%%      figure(1);
%%      scatter(objectives(:,1),objectives(:,2));
%%
%% Some concrete examples can be found in the file 'testPSO.m'.