function [ c ] = correlation_signal( I,sref ,aff)

[nz,nx,nt]=size(I);

c=zeros(nz,nx);

A0=squeeze(sref);

A0=A0-mean(A0); 
A0=A0/norm(A0);

for iz=1:nz
    for ix=1:nx
        s=squeeze(I(iz,ix,:));
         s=s-mean(s);
         s=s/norm(s);
      
c(iz,ix)=sum(s.*A0);
    end
end

if aff
 figure(105); imagesc(c); colormap jet; caxis([-0.5 0.5]);
end


end

