function [particles,velocities] = psoCreation(psoopt)
    particlesize = psoopt.ParticleSize;
    featuresize = psoopt.Featuresize;
    LB = psoopt.LB; 
    UB = psoopt.UB;
    IntCon = psoopt.IntCon;

    LBbase = meshgrid(LB,1:particlesize);
    UBbase = meshgrid(UB,1:particlesize);
    particles = LBbase + (UBbase - LBbase).*rand(particlesize,featuresize);
    velocities = 0.5*(UBbase - LBbase).*rand(particlesize,featuresize);
    
    if ~isempty(IntCon)
        intparticles = floor(particles(:,IntCon));
        intparticles = intparticles + (rand(particlesize,length(IntCon)) > 0.5);

        UBbase = UBbase(:,IntCon); LBbase = LBbase(:,IntCon);
        pos = find(intparticles>UBbase);
        intparticles(pos) = UBbase(pos);
        pos = find(intparticles<LBbase);
        intparticles(pos) = LBbase(pos);

        particles(:,IntCon) = intparticles;
    end
end