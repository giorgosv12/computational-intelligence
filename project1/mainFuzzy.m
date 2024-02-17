%Vellios Georgios Serafeim AEM:9471
clc;
close all;
clear;
FLC = readfis('FLController');


%fuzzyLogicDesigner(FLC);
out = evalfis([-0.33 -0.33], FLC)
figure(2)
gensurf(FLC)