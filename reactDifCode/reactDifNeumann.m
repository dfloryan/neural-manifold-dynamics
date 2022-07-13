function rhs = reactDifNeumann(t, z, beta, d1, d2, h, n)
% Pre-allocate arrays for du/dt and dv/dt
dudt = zeros(n, n);
dvdt = zeros(n, n);

% Extract u and v, reshape into matrices
u = reshape(z(1:n*n), n, n);
v = reshape(z(n*n + 1:2*n*n), n, n);

% Apply homogeneous Neumann boundary conditions
u(1, 2:n - 1) = (4/3)*u(2, 2:n - 1) - (1/3)*u(3, 2:n - 1); % bottom
u(n, 2:n - 1) = (4/3)*u(n - 1, 2:n - 1) - (1/3)*u(n - 2, 2:n - 1); % top
u(:, 1) = (4/3)*u(:, 2) - (1/3)*u(:, 3); % left
u(:, n) = (4/3)*u(:, n - 1) - (1/3)*u(:, n - 2); % right
v(1, 2:n - 1) = (4/3)*v(2, 2:n - 1) - (1/3)*v(3, 2:n - 1); % bottom
v(n, 2:n - 1) = (4/3)*v(n - 1, 2:n - 1) - (1/3)*v(n - 2, 2:n - 1); % top
v(:, 1) = (4/3)*v(:, 2) - (1/3)*v(:, 3); % left
v(:, n) = (4/3)*v(:, n - 1) - (1/3)*v(:, n - 2); % right

% Interior points
uint = u(2:n - 1, 2:n - 1);
vint = v(2:n - 1, 2:n - 1);
uxx = (u(3:n, 2:n - 1) - 2*u(2:n - 1, 2:n - 1) + u(1:n - 2, 2:n - 1))/h^2;
uyy = (u(2:n - 1, 3:n) - 2*u(2:n - 1, 2:n - 1) + u(2:n - 1, 1:n - 2))/h^2;
vxx = (v(3:n, 2:n - 1) - 2*v(2:n - 1, 2:n - 1) + v(1:n - 2, 2:n - 1))/h^2;
vyy = (v(2:n - 1, 3:n) - 2*v(2:n - 1, 2:n - 1) + v(2:n - 1, 1:n - 2))/h^2;
dudt(2:n - 1, 2:n - 1) = (1 - (uint.^2 + vint.^2)).*uint + ...
                         beta*(uint.^2 + vint.^2).*vint + ...
                         d1*(uxx + uyy);
dvdt(2:n - 1, 2:n - 1) = -beta*(uint.^2 + vint.^2).*uint + ...
                         (1 - (uint.^2 + vint.^2)).*vint + ...
                         d2*(vxx + vyy);

% Stack dudt and dvdt to construct rhs vector
rhs = [dudt(:); dvdt(:)];
