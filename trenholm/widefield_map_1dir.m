function [M,I]=widefield_map_1dir(datamoy)

%% fit with gaussian and correlate
count=0;
y = zeros(size(datamoy,3),size(1:2:size(datamoy,3),2));


    for i=1:2:size(datamoy,3) %make gaussian every 5 camera frames (50 ms)
    count=count+1;
   y(:,count) = gaussmf(1:(size(datamoy,3)),[45 i]); % [ G  i ] play with width of gaussian
    
    end; clear count;


C=zeros(size(datamoy,1),size(datamoy,2),size(y,2)); %matrix for correlations

%% correlate with gaussians - this will take a long time - choose number of gaussians

for i=1:size(y,2) 
    C(:,:,i)=correlation_signal(datamoy,y(:,i),0);
end    

[M,I]=max(C,[],3);
