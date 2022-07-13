% Simulates a lambda-omega reaction-diffusion system governed by
% u_t = [1 - (u^2 + v^2)]*u + beta*(u^2 + v^2)*v + d_1*(u_xx + u_yy), 
% v_t = -beta*(u^2 + v^2)*u + [1 - (u^2 + v^2)]*v + d_2*(v_xx + v_yy),
% with homogeneous Neumann boundary conditions. 

clear all
clc

% Input parameters
beta = 1; % coefficient for nonlinear term
d1 = 0.1; % first diffusion coefficient
d2 = 0.1; % second diffusion coefficient
n = 101; % number of gridpoints in each direction
L = 20; % length of domain in each direction
t = 0:0.05:500; % time

% Construct spatial grid
x = linspace(-L/2, L/2, n);
y = linspace(-L/2, L/2, n);
h = x(2) - x(1); % grid spacing
[x, y] = meshgrid(x, y);

% Set initial condition
r = sqrt(x.^2 + y.^2);
u = tanh(r.*cos(angle(x + 1i*y) - r));
v = tanh(r.*sin(angle(x + 1i*y) - r));

% Integrate forward in time
z = [u(:); v(:)]; % stack u and v into a single state vector
% z = z(end, :)';
[t, z] = ode45(@(t, z) reactDifNeumann(t, z, beta, d1, d2, h, n), t, z);

% Extract u and v
u = z(:, 1:n*n);
v = z(:, n*n + 1:2*n*n);

% Set homogeneous Neumann boundary conditions
for i=1:length(t)
    utemp = reshape(u(i, :), n, n);
    vtemp = reshape(v(i, :), n, n);
    utemp(1, 2:n - 1) = (4/3)*utemp(2, 2:n - 1) - (1/3)*utemp(3, 2:n - 1); % bottom
    utemp(n, 2:n - 1) = (4/3)*utemp(n - 1, 2:n - 1) - (1/3)*utemp(n - 2, 2:n - 1); % top
    utemp(:, 1) = (4/3)*utemp(:, 2) - (1/3)*utemp(:, 3); % left
    utemp(:, n) = (4/3)*utemp(:, n - 1) - (1/3)*utemp(:, n - 2); % right
    vtemp(1, 2:n - 1) = (4/3)*vtemp(2, 2:n - 1) - (1/3)*vtemp(3, 2:n - 1); % bottom
    vtemp(n, 2:n - 1) = (4/3)*vtemp(n - 1, 2:n - 1) - (1/3)*vtemp(n - 2, 2:n - 1); % top
    vtemp(:, 1) = (4/3)*vtemp(:, 2) - (1/3)*vtemp(:, 3); % left
    vtemp(:, n) = (4/3)*vtemp(:, n - 1) - (1/3)*vtemp(:, n - 2); % right
    u(i, :) = utemp(:);
    v(i, :) = vtemp(:);
end
z = [u, v];

% Save data
save('reactDifNeumannData.mat', 'beta', 'd1', 'd2', 't', 'x', 'y', 'z')
