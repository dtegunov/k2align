checkers=zeros(7676,7420);
checkers(1:3837,1:3709)=1;
checkers(3838:end,3710:end)=1;
stack=[];
for s=1:10
    stack(:,:,s)=uint8(round(tom_shift(checkers,[s-1,s-1])));
end;
stack=uint8(stack);
tom_rawwrite('checkerstest.dat',stack,'l');