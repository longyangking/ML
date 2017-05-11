function [globalbestparticles,globalbestobjectives] = PSO(psoopt)
    % Read PSO options
    func = psoopt.Fitnessfunction;
    featuresize = psoopt.Featuresize;
    targetsize = psoopt.Targetsize;
    %numofparallel = psoopt.NumOfParallel;
    %LB = psoopt.LB; 
    %UB = psoopt.UB;
    %IntCon = psoopt.IntCon;
    constraintfunction = psoopt.Constraintfunction;
    particlesize = psoopt.ParticleSize;
    initialparticles = psoopt.InitialParticles;
    initialvelocities = psoopt.InitialVelocities;
    %initiallocalbestparticles = psoopt.InitialLocalBestParticles;
    %initialglobalbestparticles = psoopt.InitialGlobalBestParticles;
    C1 = psoopt.C1;
    C2 = psoopt.C2;
    w = psoopt.w;
    %mutationrate = psoopt.MutationRate;
    %stalliterationlimit = psoopt.StallIterationLimit;
    %stalltimelimit = psoopt.StallTimeLimit;
    %tolerancefitness = psoopt.ToleranceFitness;
    %timelimit = psoopt.Timelimit;
    maxiteration = psoopt.MaxIteration;
    verbose = psoopt.Verbose;
    creationfunction = psoopt.Creationfunction;
    fitnessscalefunction = psoopt.Fitnessscalefunction;
    distancefunction = psoopt.Distancefunction;
    mutationfunction = psoopt.Mutationfunction;
    hyperfunction = psoopt.Hyperfunction;
    
    % Initiate the PSO structure
    [particles,velocities] = creationfunction(psoopt);
    if ~isempty(initialparticles)
        particles = initialparticles;
    end
    
    if ~isempty(initialvelocities)
        velocities = initialvelocities;
    end
    fitness = zeros(particlesize,targetsize);
    objectives = zeros(particlesize,targetsize);
    localbestparticles = []; globalbestparticles = [];
    localbestfitness = []; globalbestfitness = [];
    localbestobjectives = []; globalbestobjectives = [];
    % Main Loop
    iter = 0; % iter = 0 -> initiate local-best and group-best
    while iter <= maxiteration 
        if iter == 0
            velocities = w*velocities;
            newparticles = particles + velocities;

            combineparticles = [particles; newparticles];
            objectives = func(combineparticles);
            size(objectives)
            size(repmat(constraintfunction(combineparticles),1,targetsize))
            fitness = objectives + repmat(constraintfunction(combineparticles),1,targetsize); % To deal with constraint

            scaledfitness = fitnessscalefunction(fitness);
            [ranks,distances] = distancefunction(scaledfitness);

            % Initate Local-best
            status = (ranks(1:particlesize) > ranks(particlesize+1:2*particlesize)) | ...
                        (ranks(1:particlesize) == ranks(particlesize+1:2*particlesize) & distances(1:particlesize) < distances(particlesize+1:2*particlesize));
            pos = find(status);
            localbestparticles = particles;
            localbestparticles(pos,:) = newparticles(pos,:);

            particlesfitness = fitness(1:particlesize,:);
            newparticlesfitness = fitness(particlesize+1:2*particlesize,:);
            localbestfitness = particlesfitness;
            localbestfitness(pos,:) = newparticlesfitness(pos,:);

            particlesobjectives = objectives(1:particlesize,:);
            newparticlesobjectives = objectives(particlesize+1:2*particlesize,:);
            localbestobjectives = particlesobjectives;
            localbestobjectives(pos,:) = newparticlesobjectives(pos,:);
            
            % Initiate Global-best
            pos = find(ranks==0);
            globalbestparticles = combineparticles(pos,:); 
            globalbestfitness = fitness(pos,:);
            globalbestobjectives = objectives(pos,:);

            particles = newparticles;
            
            hyperfunction(globalbestparticles,globalbestobjectives,psoopt,iter);
            iter = iter + 1;

            if verbose
                fprintf('%dth update with number of Pareto fronts: %d ...\n',iter,size(globalbestparticles,1));
            end
            continue;
        end

        [globalcount,temp] = size(globalbestparticles);

        randC = rand(particlesize,2);
        randglobal = randi(globalcount,1,particlesize);
        velocities = w*velocities ...
                + C1*repmat(randC(:,1),1,featuresize).*(localbestparticles - particles) ...
                + C2*repmat(randC(:,2),1,featuresize).*(globalbestparticles(randglobal,:) - particles);

        particles = particles + velocities;
        particles = mutationfunction(particles,psoopt);

        objectives = func(particles);
        fitness = objectives + repmat(constraintfunction(particles),1,targetsize);  % To deal with constraint

        totalfitness = [fitness; localbestfitness; globalbestfitness];
        totalobjectives = [objectives; localbestobjectives; globalbestobjectives];
        scaledfitness = fitnessscalefunction(totalfitness);
        [ranks,distances] = distancefunction(scaledfitness);

        % Update Local-best
        status = (ranks(1:particlesize) > ranks(particlesize+1:2*particlesize)) | ...
                        (ranks(1:particlesize) == ranks(particlesize+1:2*particlesize) & distances(1:particlesize) < distances(particlesize+1:2*particlesize));
        pos = find(status);
        localbestparticles(pos,:) = particles(pos,:);
        localbestfitness(pos,:) = totalfitness(pos,:);
        localbestobjectives(pos,:) = totalobjectives(pos,:);
            
        % Initiate Global-best
        pos = find(ranks==0);
        combineparticles = [particles; localbestparticles; globalbestparticles];
        globalbestparticles = combineparticles(pos,:);
        globalbestfitness = totalfitness(pos,:); 
        globalbestobjectives = totalobjectives(pos,:);

        hyperfunction(globalbestparticles,globalbestobjectives,psoopt,iter);
        iter = iter + 1;

        if verbose
            fprintf('%dth update with number of Pareto fronts: %d ...\n',iter,size(globalbestparticles,1));
        end
    end

end