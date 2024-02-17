%Vellios Georgios Serafeim AEM:9471
clc;
clear;
close all;

Kp=1.3;

zeros = -0.3;
poles = 0;
gain = Kp;
Gc = zpk(zeros, poles, gain);

Gp = zpk([], [-0.1, -10], 25);

sys_open_loop = Gc*Gp;

figure(1);
rlocus(sys_open_loop)
sys = feedback(sys_open_loop, 1);
figure(2);
step(sys);
s = stepinfo(sys)
disp("Rise Time:")
disp(s.RiseTime)
disp("Overshoot (%):")
disp(s.Overshoot)

k = 25*Kp
Kp
Kl = -(zeros*Kp)




