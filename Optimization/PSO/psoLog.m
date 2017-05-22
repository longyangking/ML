% Author: Yang Long
%
% E-mail: longyang_123@yeah.net

function psoLog(globalbestparticles,globalbestobjectives,psoopt,iter)
    directory = 'PSO_log';
    if iter == 0
        if ~exist(directory)
            mkdir(directory);
        else
            rmdir(directory,'s');   % Renew the log
            mkdir(directory);
        end
        fid = fopen(strcat(directory,'\info.txt'),'a');
        % Record the time
        fprintf(fid,'Start Time: %s\n\n',datestr(now(),'dd-mmm-yyyy HH:MM:SS'));
        % Record PSO struct
        infonames = fieldnames(psoopt);
        for index = 1:length(infonames)
            if isempty(strfind(infonames{index},'function'))
                fprintf(fid,'%s: %s\n',infonames{index},num2str(getfield(psoopt,infonames{index}))); 
            end
        end
        fclose(fid);
        return
    end

    fid = fopen(strcat(directory,'\',num2str(iter),'.txt'),'a');
    [globalcount,featuresize] = size(globalbestparticles);
    for index = 1:globalcount
        fprintf(fid,'%d\n',index);
        fprintf(fid,'%10.5f',globalbestparticles(index,:)); fprintf(fid,'\n');
        fprintf(fid,'%10.5f',globalbestobjectives(index,:)); fprintf(fid,'\n');
    end
    fclose(fid);
end