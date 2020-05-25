a = 5.3256 ;
b = 6 ;

A=[1 2; 3 4 ; 5 6];
rand(3,3);
randn(1,3);
eye(4) %4x4 identity matrix
disp(a == b)  
disp(sprintf('2 decimals: %0.2f', a))

sz = size(A);
size(A,1); % number of rows
format long ;
format short;

%load featuresX.dat
%load('featuresX.dat')

%who shows variables in the current scope
%clear featuresX %clears the data
%V = featuresX(1:10)
%save hello.mat v; %saves the v as file shown
%save hello.txt v -ascii
%A([1 3],:) gives A's ffirst and third row with all columns
%a(:,2) = [10;11;12]
%A = [A,[10;11;12]]
%A(:) put all elements of A into a single vector
%element wise multiply matrix = A.*B
%A.^2
%1 ./ V

% [val ,ind] = max(A)  %return values and index of max values
a= [0.4 , 1.5 ,15,5.9];
%floor(a) % rounds down
%ceil(a) %rounds down
%max(A,[],1) return max numbers in columns
%max(A,[],2) return max numbers in rows
%sum(A,1) sums all columns
%pinv(A) inverse of A
 
#plotting data
t=[0:0.01:0.98];
y1=sin(2*pi*4*t);
y2=cos(2*pi*4*t);
plot(t,y1)
hold on;
plot(t,y2,'r');
xlabel('Time')
ylabel('Value')
legend('sin','cos')
title('My Plot')
##cd 'C:\Users\User\Coursera'; 
##print -dpng 'myFirstPlot.png'
##figure();plot(t,y1);
figure();
subplot(2,1,1);plot(t,y1);
subplot(2,1,2);plot(t,y2);
%axis([0.5 1 -1 1])
v=ones(8,1)
for i=1:10,
  v(i)=2^i;
end

%addpath('C:\Users\User\Desktop')
 
##[t1,t2]=sq_2(6) function call




























