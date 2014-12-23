rng(123);
pattern=randn(1919,1865).*255;
stack=zeros([size(pattern) 40]);
for s=1:40
    stack(:,:,s)=uint8(round(tom_shift(pattern,[s-1,s-1].*3)));
end;
stack=uint8(stack);
tom_rawwrite('teststack.dat',stack,'l');
tom_emwrite('teststack.em',single(stack));
tom_mrcwrite(single(stack),'name','teststack.mrc','style','classic');