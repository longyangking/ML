% Author: Yang Long
%
% E-mail: longyang_123@yeah.net

function newparticles = psoMutation(particles,psoopt)
    mutationrate = psoopt.MutationRate;
    LB = psoopt.LB; 
    UB = psoopt.UB;
    IntCon = psoopt.IntCon;

    [particlesize,featuresize] = size(particles);
    newparticles = particles;
    LBbase = meshgrid(LB,1:particlesize);
    UBbase = meshgrid(UB,1:particlesize);
    
    Preal = mutationrate;
    Pint = 2*Preal;
    P = Preal*ones(particlesize,featuresize);
    if ~isempty(IntCon)
        P(:,IntCon) = Pint;
    end
    S = rand(particlesize,featuresize);
    pos = S>P;
    T = rand(particlesize,featuresize);
    newparticles(pos) = LBbase(pos) + (UBbase(pos) - LBbase(pos)).*T(pos);
    
    if ~isempty(IntCon)
        intparticles = floor(newparticles(:,IntCon));
        intparticles = intparticles + 1*(rand(particlesize,length(IntCon)) > 0.5);

        UBbase = UBbase(:,IntCon); LBbase = LBbase(:,IntCon);
        pos = find(intparticles>UBbase);
        intparticles(pos) = UBbase(pos);
        pos = find(intparticles<LBbase);
        intparticles(pos) = LBbase(pos);

        newparticles(:,IntCon) = intparticles;
    end
end