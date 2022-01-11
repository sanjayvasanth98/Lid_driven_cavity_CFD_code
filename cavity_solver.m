function [rL1norm, rL2norm, rLinfnorm] = cavity_solver()
tic   %begin timer function
%--- Variables for file handling ---
%--- All files are globally accessible ---

global fp1 % For output of iterative residual history
global fp2 % For output of field data (solution)
%   global fp3 % For writing the restart file
%   global fp4 % For reading the restart file
%   global fp5 % For output of final DE norms (only for MMS)
%$$$$$$   global fp6 % For debug: Uncomment for debugging.

global imax jmax neq nmax
global zero tenth sixth fifth fourth third half one two three four six
global iterout imms isgs irstr ipgorder lim cfl Cx Cy toler rkappa Re pinf uinf rho rhoinv xmin xmax ymin ymax Cx2 Cy2 fsmall
global rlength rmu vel2ref dx dy rpi phi0 phix phiy phixy apx apy apxy fsinx fsiny fsinxy

%**Use these variables cautiously as these are globally accessible from all functions.**

global u;         % Solution vector [p, u, v]^``T at each node
global uold;      % Previous (old) solution vector
global s;         % Source term
global dt;        % Local time step at each node
global artviscx;  % Artificial viscosity in x-direction
global artviscy;  % Artificial viscosity in y-direction
global ummsArray; % Array of umms values (funtion umms evaluated at all nodes)

%************ Following are fixed parameters for array sizes *************
imax = 129;   	% Number of points in the x-direction (use odd numbers only)
jmax = 129;   	% Number of points in the y-direction (use odd numbers only)
neq = 3;       % Number of equation to be solved ( = 3: mass, x-mtm, y-mtm)
%********************************************
%***** All  variables declared here. **
%**** These variables SHOULD not be changed *
%********* by the program once set. *********
%********************************************
%**** The variables declared "" CAN ****
%** not be changed by the program once set **
%********************************************

%--------- Numerical constants --------
zero   = 0.0;
tenth  = 0.1;
sixth  = 1.0/6.0;
fifth  = 0.2;
fourth = 0.25;
third  = 1.0/3.0;
half   = 0.5;
one    = 1.0;
two    = 2.0;
three  = 3.0;
four   = 4.0;
six    = 6.0;

%--------- User sets inputs here  --------

nmax = 500000;        % Maximum number of iterations
iterout = 1000;       % Number of time steps between solution output
imms = 0;             % Manufactured solution flag: = 1 for manuf. sol., = 0 otherwise
isgs = 1;             % Symmetric Gauss-Seidel  flag: = 1 for SGS, = 0 for point Jacobi
irstr = 0;            % Restart flag: = 1 for restart (file 'restart.in', = 0 for initial run
ipgorder = 0;         % Order of pressure gradient: 0 = 2nd, 1 = 3rd (not needed)
lim = 1;              % variable to be used as the limiter sensor (= 1 for pressure)

cfl  = 0.5;      % CFL number used to determine time step
Cx = 0.007;     	% Parameter for 4th order artificial viscosity in x
Cy = 0.007;      	% Parameter for 4th order artificial viscosity in y
toler = 1.e-10; 	% Tolerance for iterative residual convergence
rkappa = 0.1;   	% Time derivative preconditioning constant
Re = 100.0;      	% Reynolds number = rho*Uinf*L/rmu
pinf = 0.801333844662; % Initial pressure (N/m^2) -> from MMS value at cavity center
uinf = 1.0;      % Lid velocity (m/s)
rho = 1.0;       % Density (kg/m^3)
xmin = 0.0;      % Cavity dimensions...: minimum x location (m)
xmax = 0.05;   	%                       maximum x location (m)
ymin = 0.0;      %                       maximum y location (m)
ymax = 0.05;   	%                       maximum y location (m)
Cx2 = 0.0;       % Coefficient for 2nd order damping (not required)
Cy2 = 0.0;     	% Coefficient for 2nd order damping (not required)
fsmall = 1.e-20; % small parameter
filename = sprintf('./imax%03d_CFL%4.2f_C%4.2f_K%4.2f_Re%03d',imax,cfl,Cx,rkappa,Re);

%-- Derived input quantities (set by function 'set_derived_inputs' called from main)----

rhoinv =  -99.9; 	% Inverse density, 1/rho (m^3/kg)
rlength = -99.9;  	% Characteristic length (m) [cavity width]
rmu = -99.9;  		% Viscosity (N*s/m^2)
vel2ref = -99.9;  	% Reference velocity squared (m^2/s^2)
dx = -99.9; 		%	 Delta x (m)
dy = -99.9;  		% Delta y (m)
rpi = -99.9; 		% Pi = 3.14159... (defined below)

%-- constants for manufactured solutions ----
phi0 = [0.25, 0.3, 0.2];          % MMS constant
phix = [0.5, 0.15, 1.0/6.0];      % MMS amplitude constant
phiy = [0.4, 0.2, 0.25];          % MMS amplitude constant
phixy = [1.0/3.0, 0.25, 0.1];     % MMS amplitude constant
apx = [0.5, 1.0/3.0, 7.0/17.0]; 	% MMS frequency constant
apy = [0.2, 0.25, 1.0/6.0];         % MMS frequency constant
apxy = [2.0/7.0, 0.4, 1.0/3.0];     % MMS frequency constant
fsinx = [0.0, 1.0, 0.0];            % MMS constant to determine sine vs. cosine
fsiny = [1.0, 0.0, 0.0];            % MMS constant to determine sine vs. cosine
fsinxy = [1.0, 1.0, 0.0];           % MMS constant to determine sine vs. cosine
% Note: fsin = 1 means the sine function
% Note: fsin = 0 means the cosine function
% Note: arrays here refer to the 3 variables

%************************************************************************
%      						Main Function
%************************************************************************
%----- Looping indices --------

% i = 0;                       % i index (x direction)
% j = 0;                       % j index (y direction)
% k = 0;                       % k index (# of equations)
% n = 0;	                   % Iteration number index

%conv = -99.9 ; % Minimum of iterative residual norms from three equations


%--------- Solution variables declaration --------

ninit = 0;        	% Initial iteration number (used for restart file)

%$$$$$$    u(imax,jmax,neq);         % Solution vector (p, u, v)^T at each node
%$$$$$$    uold(imax,jmax,neq);      % Previous (old) solution vector
%$$$$$$    s(imax,jmax,neq);    % Source term
%$$$$$$    dt(imax,jmax);        % Local time step at each node
%$$$$$$    artviscx(imax,jmax);  % Artificial viscosity in x-direction
%$$$$$$    artviscy(imax,jmax);  % Artificial viscosity in y-direction
% res = [0,0,0];              % Iterative residual for each equation
% resinit = [0,0,0];          % Initial iterative residual for each equation (from iteration 1)
% rL1norm = [0,0,0];          % L1 norm of discretization error for each equation
% rL2norm = [0,0,0];          % L2 norm of discretization error for each equation
% rLinfnorm = [0,0,0];        % Linfinity norm of discretization error for each equation
rtime = -99.9;         % Variable to estimate simulation time
% dtmin = 1.0e99;        % Minimum time step for a given iteration (initialized large)

% x = -99.9;       % Temporary variable for x location
% y = -99.9;       % Temporary variable for y location

% Solution variables initialization with dummy values
% for i=1:imax
%  for j=1:jmax
%      dt(i,j) = -99.9;
%      artviscx(i,j) = -99.9;
%      artviscy(i,j) = -99.9;
%    for k=1:neq
%      u(i,j,k) = -99.9;
%      uold(i,j,k) = -99.9;
%      s(i,j,k) = -99.9;
%      res(k) = -99.9;
%      resinit(k) = -99.9;
%      res(k) = -99.9;
%      rL1norm(k) = -99.9;
%      rL2norm(k) = -99.9;
%      rLinfnorm(k) = -99.9;
%    end
%  end
% end

dt = zeros(imax,jmax);
artviscx = zeros(imax,jmax);
artviscy = zeros(imax,jmax);
u = zeros(imax,jmax,neq);
uold = zeros(imax,jmax,neq);
ummsArray = zeros(imax,jmax,neq);
s = zeros(imax,jmax,neq);
res = zeros(neq,1);
resinit = zeros(neq,1);
rL1norm = zeros(neq,1);
rL2norm = zeros(neq,1);
rLinfnorm = zeros(neq,1);

dt(:,:) = -99.9;
artviscx(:,:) = -99.9;
artviscy(:,:) = -99.9;
u(:,:,:) = -99.9;
uold(:,:,:) = -99.9;
s(:,:,:) = -99.9;
res(:) = -99.9;
resinit(:) = -99.9;
% rL1norm(:) = -99.9;
% rL2norm(:) = -99.9;
% rLinfnorm(:) = -99.9;


% Debug output: Uncomment and modify if debugging
%$$$$$$ fp6 = fopen("./Debug.dat","w");
%$$$$$$ fprintf(fp6,"TITLE = \"Debug Data Data\"\n");
%$$$$$$ fprintf(fp6,"variables=\"x(m)\"\"y(m)\"\"visc-x\"\"visc-y\"\n");
%$$$$$$ fprintf(fp6, "zone T=\"n=%d\"\n",n);
%$$$$$$ fprintf(fp6, "I= %d J= %d\n",imax, jmax);
%$$$$$$ fprintf(fp6, "DATAPACKING=POINT\n");

% Set derived input quantities
set_derived_inputs();

% Set up headers for output files

output_file_headers(filename);


% Set Initial Profile for u vector
[ninit, rtime, resinit] = initial(ninit, rtime, resinit);

% Set Boundary Conditions for u
set_boundary_conditions();

% Write out inital conditions to solution file
write_output(ninit, resinit, rtime);

% Initialize Artificial Viscosity arrays to zero (note: artviscx(i,j) and artviscy(i,j)
artviscx(:,:) = zero;
artviscy(:,:) = zero;

% Evaluate Source Terms Once at Beginning
%(only interior points; will be zero for standard cavity)
compute_source_terms();

%========== Main Loop ==========
isConverged = 0;

for n = ninit:nmax
    % Calculate time step
    %dtmin = compute_time_step(dtmin);
    dtmin = compute_time_step();
    
    % Save u values at time level n (u and uold are 2D arrays)
    uold = u;
    
    if isgs==1 % ==Symmetric Gauss Seidel==
        
        % Artificial Viscosity
        Compute_Artificial_Viscosity();
        
        % Symmetric Gauss-Siedel: Forward Sweep
        SGS_forward_sweep();
        
        % Set Boundary Conditions for u
        set_boundary_conditions();
        
        % Artificial Viscosity
        Compute_Artificial_Viscosity();
        
        % Symmetric Gauss-Siedel: Backward Sweep
        SGS_backward_sweep();
        
        % Set Boundary Conditions for u
        set_boundary_conditions();
    else
        if isgs==0 % ==Point Jacobi==

            % Artificial Viscosity
            Compute_Artificial_Viscosity();

            % Point Jacobi: Forward Sweep
            point_Jacobi();

            % Set Boundary Conditions for u
            set_boundary_conditions();
        else
            fprintf('ERROR: isgs must equal 0 or 1!\n');
            return;
        end
    end
    
    % Pressure Rescaling (based on center point)
    pressure_rescaling();
    
    % Update the time
    rtime = rtime + dtmin;
    
    % Check iterative convergence using L2 norms of iterative residuals
    [res, resinit, conv] = check_iterative_convergence(n, res, resinit, ninit, rtime, dtmin);
    
    if(conv<toler)
        fprintf(fp1, '%d %e %e %e %e\n',n, rtime, res(1), res(2), res(3));
        isConverged = 1;
        break;
    end
    
    % Output solution and restart file every 'iterout' steps
    if( (mod(n,iterout)==0) )
        write_output(n, resinit, rtime);
    end
    
end  % ========== End Main Loop ==========

if isConverged == 0
    fprintf('Solution failed to converge in %d iterations!!!', nmax);
end

if isConverged == 1
    fprintf('Solution converged in %d iterations!!!', n);
end

% Calculate and Write Out Discretization Error Norms (will do this for MMS only)
[rL1norm, rL2norm, rLinfnorm] = Discretization_Error_Norms(filename);


% Discretization_Error_Norms(rL1norm, rL2norm, rLinfnorm);

% Output solution and restart file
write_output(n, resinit, rtime);

% Close open files
fclose(fp1);
fclose(fp2);
%$$$$$$   fclose(fp6); % Uncomment for debug output (

PrsMatrix = u(:,:,1);    %output arrays
uvelMatrix = u(:,:,2);
vvelMatrix = u(:,:,3);
toc  %end timer function
end

%**************************************************************************/
%*      					All Other	Functions					      */
%**************************************************************************/

%**************************************************************************
%**************************************************************************
function set_derived_inputs(~)
global imax jmax
global one
global Re uinf rho rhoinv xmin xmax ymin ymax
global rlength rmu vel2ref dx dy rpi

rhoinv = one/rho;                            % Inverse density, 1/rho (m^3/kg) */
rlength = xmax - xmin;                       % Characteristic length (m) [cavity width] */
rmu = rho*uinf*rlength/Re;                   % Viscosity (N*s/m^2) */
vel2ref = uinf*uinf;                         % Reference velocity squared (m^2/s^2) */
dx = (xmax - xmin)/(imax - 1);          % Delta x (m) */
dy = (ymax - ymin)/(jmax - 1);          % Delta y (m) */
rpi = acos(-one);                            % Pi = 3.14159... */
fprintf('rho,V,L,mu,Re: %f %f %f %f %f\n',rho,uinf,rlength,rmu,Re);
end

%************************************************************************
function output_file_headers(filename)

% Uses global variable(s): imms, fp1, fp2
% Note: The vector of primitive variables is:
%               u = [p, u, v]^T
% Set up output files (history and solution)

global imms fp1 fp2

fp1 = fopen([filename,'_history.dat'],'w');
fprintf(fp1,'TITLE = "Cavity Iterative Residual History"\n');
fprintf(fp1,'variables="Iteration""Time(s)""Res1""Res2""Res3"\n');

fp2 = fopen([filename,'_cavity.dat'],'w');
fprintf(fp2,'TITLE = "Cavity Field Data"\n');
if (imms==1)
    
    fprintf(fp2,'variables="x(m)""y(m)""p(N/m^2)""u(m/s)""v(m/s)"');
    fprintf(fp2,'"p-exact""u-exact""v-exact""DE-p""DE-u""DE-v"\n');
    
else
    
    if (imms==0)
        
        fprintf(fp2,'variables="x(m)""y(m)""p(N/m^2)""u(m/s)""v(m/s)"\n');
        
    else
        
        fprintf('ERROR! imms must equal 0 or 1!!!\n');
        return;
    end
end

% Header for Screen Output
fprintf('Iter. Time (s)   dt (s)      Continuity    x-Momentum    y-Momentum\n');

end
%************************************************************************
function [ninit, rtime, resinit] = initial(ninit, rtime, resinit)
%
%Uses global variable(s): zero, one, irstr, imax, jmax, neq, uinf, pinf
%To modify: ninit, rtime, resinit, u, s

% i                        % i index (x direction)
% j                        % j index (y direction)
% k                        % k index (# of equations)
% x        % Temporary variable for x location
% y        % Temporary variable for y location

% This subroutine sets inital conditions in the cavity
% Note: The vector of primitive variables is:
%              u = (p, u, v)^T

global zero one irstr imax jmax neq uinf pinf xmax xmin ymax ymin
global u s ummsArray

if (irstr==0)   % Starting run from scratch
    ninit = 1;          % set initial iteration to one
    rtime = zero;       % set initial time to zero
    for k = 1:neq
        resinit(k) = one;
    end
    for j = 1:jmax
        for i = 1:imax
            u(i,j,1) = pinf;
            u(i,j,2) = zero;
            u(i,j,3) = zero;
            s(i,j,1) = zero;
            s(i,j,2) = zero;
            s(i,j,3) = zero;
        end
        u(i,jmax,2) = uinf; % Initialize lid (top) to freestream velocity
    end
else
    if (irstr==1)  % Restarting from previous run (file 'restart.in')
        fp4 = fopen('./restart.in','r'); % Note: 'restart.in' must exist!
        if (fp4==NULL)
            fprintf('Error opening restart file. Stopping.\n');
            return;
        end
        fscanf(fp4, '%d %lf', ninit, rtime); % Need to known current iteration # and time value
        fscanf(fp4, '%lf %lf %lf', resinit(0), resinit(1), resinit(2)); % Needs initial iterative residuals for scaling
        for j=1:jmax
            for i=1:imax
                x = (xmax - xmin)*(i-1)/(imax - 1); %Local Definition of x and y(may not be needed)
                y = (ymax - ymin)*(j-1)/(jmax - 1);
                fscanf(fp4, '%lf %lf %lf %lf %lf', x, y, u(i,j,1), u(i,j,2), u(i,j,3));
            end
        end
        ninit = ninit + 1;
        fprintf('Restarting at iteration %d\n', ninit);
        fclose(fp4);
    else
        printf('ERROR: irstr must equal 0 or 1!\n');
        return;
    end
end

%initialize the ummsArray with values computed with umms function
for j=1:jmax
    for i=1:imax
        for k=1:neq
            x = (xmax - xmin)*(i-1)/(imax - 1);
            y = (ymax - ymin)*(j-1)/(jmax - 1);
            ummsArray(i,j,k) = umms(x,y,k);
        end
    end
end
end
%************************************************************************
function set_boundary_conditions(~)
%
%Uses global variable(s): imms
%To modify: u (via other functions: bndry() and bndrymms())

global imms

% This subroutine determines the appropriate BC routines to call
if (imms==0)
    bndry();
else
    if (imms==1)
        bndrymms();
    else
        printf('ERROR: imms must equal 0 or 1!\n');
        return;
    end
end
end
%************************************************************************
function bndry(~)
%
%Uses global variable(s): zero, one, two, half, imax, jmax, uinf
%To modify: u

% i                        % i index (x direction)
% j                        % j index (y direction)

global imax jmax uinf
global u

% This applies the cavity boundary conditions


% !************************************************************** */
% !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
% !************************************************************** */
%bottom wall
for i=2:imax-1
    j =1;
    u(i,j,1) = 2*u(i,j+1,1)-u(i,j+2,1);
    u(i,j,2) = 0;
    u(i,j,3) = 0;
    
end

%side wall - left

for j = 2:jmax-1
    i = 1;
    
    u(i,j,1) = 2*u(i+1,j,1)-u(i+2,j,1);
    u(i,j,2) = 0;
    u(i,j,3) = 0;
    
end

%side wall - right

for j = 2:jmax-1
    i = imax;
   
    u(i,j,1) = 2*u(i-1,j,1)-u(i-2,j,1);
    u(i,j,2) = 0;
    u(i,j,3) = 0;
   
end

%top wall

for i=2:imax-1
    j =jmax;
    u(i,j,1) = 2*u(i,j-1,1)-u(i,j-2,1);
    u(i,j,2) = uinf;
    u(i,j,3) = 0;
    
    
end
%pressure BC at corners
%  u(1,1,1) = 1/3*(u(1,2,1)+u(2,2,1)+u(2,1,1)); %bottom left corner
%  u(imax,1,1) = 1/3*(u(imax,2,1)+u(imax-1,2,1)+u(imax-1,1,1)); %bottom right corner
%  u(imax,jmax,1) = 1/3*(u(imax-1,jmax,1)+u(imax-1,jmax-1,1)+u(imax,jmax-1,1)); %top right corner
%  u(1,jmax,1) = 1/3*(u(1,jmax-1,1)+u(2,jmax-1,1)+u(2,jmax,1));%top left corner

%bottom left corner
ux = 2*u(2,1,1)-u(3,1,1);
uy = 2*u(1,2,1)-u(1,3,1);
u(1,1,1) = 0.5*(ux+uy);
 %bottom right corner
ux = 2*u(imax-1,1,1)-u(imax-2,1,1);
uy = 2*u(imax,2,1)-u(imax,3,1);
u(imax,1,1) = 0.5*(ux+uy);
%top right corner
ux = 2*u(imax-1,jmax,1)-u(imax-2,jmax,1);
uy = 2*u(imax,jmax-1,1)-u(imax,jmax-2,1);
u(imax,jmax,1) = 0.5*(ux+uy);
%top left corner
ux = 2*u(2,jmax,1)-u(3,jmax,1);
uy = 2*u(1,jmax-1,1)-u(1,jmax-2,1);
u(1,jmax,1) = 0.5*(ux+uy);
 
 %u-vel at corners
 u(1,1,2) = 0; %bottom left corner
 u(imax,1,2) = 0 ;%bottom right corner
 u(imax,jmax,2) = uinf ;%top right corner
 u(1,jmax,2) = uinf; %top left corner
 
 %v-vel at corners
 u(1,1,3) = 0; %bottom left corner
 u(imax,1,3) = 0 ;%bottom right corner
 u(imax,jmax,3) = 0 ;%top right corner
 u(1,jmax,3) = 0; %top left corner
 
end
%************************************************************************
function bndrymms(~)
%
%Uses global variable(s): two, imax, jmax, neq, xmax, xmin, ymax, ymin, rlength
%To modify: u
% i                        % i index (x direction)
% j                        % j index (y direction)
% k                        % k index (# of equations)
% x        % Temporary variable for x location
% y        % Temporary variable for y location
% This applies the cavity boundary conditions for the manufactured solution

global two imax jmax neq
global u ummsArray

% Side Walls
for j = 2:jmax-1
    i = 1;
    for k = 1:neq
        u(i,j,k) = ummsArray(i,j,k);
    end
    u(1,j,1) = two*u(2,j,1) - u(3,j,1);    % 2nd Order BC
    %    u(1,j,1) = u(2,j,1);                  % 1st Order BC
    
    i=imax;
    for k = 1:neq
        u(i,j,k) = ummsArray(i,j,k);
    end
    u(imax,j,1) = two*u(imax-1,j,1) - u(imax-2,j,1);   % 2nd Order BC
    %	u(imax,j,1) = u(imax-1,j,1);                       % 1st Order BC
end

% Top/Bottom Walls
for i=1:imax
    j = 1;
    for k = 1:neq
        u(i,j,k) = ummsArray(i,j,k);
    end
    u(i,1,1) = two*u(i,2,1) - u(i,3,1);   % 2nd Order BC
    %$$$$$$     u(i,1,1) = u(i,2,1);            % 1st Order BC
    
    j = jmax;
    for k = 1:neq
        u(i,j,k) = ummsArray(i,j,k);
    end
    u(i,jmax,1) = two*u(i,jmax-1,1) - u(i,jmax-2,1);   % 2nd Order BC
    %$$$$$$     u(i,jmax,1) = u(i,jmax-1,1);              % 1st Order BC
end
end
%************************************************************************
function [ummstmp] = umms( x, y, k)
%
%Uses global variable(s): one, rpi, rlength
%Inputs: x, y, k
%To modify: <none>
%Returns: umms

% ummstmp; % Define return value for umms as % precision

% termx       % Temp variable
% termy       % Temp variable
% termxy      % Temp variable
% argx        % Temp variable
% argy        % Temp variable
% argxy       % Temp variable

% This function returns the MMS exact solution

global one rpi rlength
global phi0 phix phiy phixy apx apy apxy fsinx fsiny fsinxy

argx = apx(k)*rpi*x/rlength;
argy = apy(k)*rpi*y/rlength;
argxy = apxy(k)*rpi*x*y/rlength/rlength;
termx = phix(k)*(fsinx(k)*sin(argx)+(one-fsinx(k))*cos(argx));
termy = phiy(k)*(fsiny(k)*sin(argy)+(one-fsiny(k))*cos(argy));
termxy = phixy(k)*(fsinxy(k)*sin(argxy)+(one-fsinxy(k))*cos(argxy));

ummstmp = phi0(k) + termx + termy + termxy;
end
%************************************************************************
function write_output( n,  resinit,  rtime)
%
%Uses global variable(s): imax, jmax, new, xmax, xmin, ymax, ymin, rlength, imms
%Uses global variable(s): ninit, u, dt, resinit, rtime
%To modify: <none>
%Writes output and restart files.

% i                        % i index (x direction)
% j                        % j index (y direction)
% k                        % k index (# of equations)

% x        % Temporary variable for x location
% y        % Temporary variable for y location

global imax jmax xmax xmin ymax ymin imms
global u ummsArray
global fp2 fp3

% Field output
fprintf(fp2, 'zone T="n=%d"\n',n);
fprintf(fp2, 'I= %d J= %d\n',imax, jmax);
fprintf(fp2, 'DATAPACKING=POINT\n');

if (imms==1)
    for j=1:jmax
        for i=1:imax
            x = (xmax - xmin)*(i-1)/(imax - 1);
            y = (ymax - ymin)*(j-1)/(jmax - 1);
            fprintf(fp2,'%e %e %e %e %e %e %e %e %e %e %e\n', x, y, ...
                u(i,j,1), u(i,j,2), u(i,j,3), ummsArray(i,j,1), ummsArray(i,j,2), ummsArray(i,j,3), ...
                (u(i,j,1)-ummsArray(i,j,1)), (u(i,j,2)-ummsArray(i,j,2)), (u(i,j,3)-ummsArray(i,j,3)));
        end
    end
else
    if (imms==0)
        for j=1:jmax
            for i=1:imax
                x = (xmax - xmin)*(i-1)/(imax - 1);
                y = (ymax - ymin)*(j-1)/(jmax - 1);
                fprintf(fp2,'%e %e %e %e %e\n', x, y, ...
                    u(i,j,1), u(i,j,2), u(i,j,3));
            end
        end
    else
        fprintf('ERROR: imms must equal 0 or 1!\n');
        return;
    end
end

% Restart file: overwrites every 'iterout' iteration
fp3 = fopen('./restart.out','w');
fprintf(fp3,'%d %e\n', n, rtime);
fprintf(fp3,'%e %e %e\n', resinit(1), resinit(2), resinit(3));
for j=1:jmax
    for i=1:imax
        x = (xmax - xmin)*(i-1)/(imax - 1);
        y = (ymax - ymin)*(j-1)/(jmax - 1);
        fprintf(fp3,'%e %e %e %e %e\n', x, y, ...
            u(i,j,1), u(i,j,2), u(i,j,3));
    end
end
fclose(fp3);
end
%************************************************************************
function compute_source_terms(~)
%
%Uses global variable(s): imax, jmax, imms, rlength, xmax, xmin, ymax, ymin
%To modify: s (source terms)

% i                        % i index (x direction)
% j                        % j index (y direction)

% x        % Temporary variable for x location
% y        % Temporary variable for y location

% Evaluate Source Terms Once at Beginning (only %erior po%s; will be zero for standard cavity)

global imax jmax imms xmax xmin ymax ymin
global s

for j=2:jmax-1
    for i=2:imax-1
        x = (xmax - xmin)*(i-1)/(imax - 1);
        y = (ymax - ymin)*(j-1)/(jmax - 1);
        s(i,j,1) = (imms)*srcmms_mass(x,y);
        s(i,j,2) = (imms)*srcmms_xmtm(x,y);
        s(i,j,3) = (imms)*srcmms_ymtm(x,y);
    end
end
end
%************************************************************************
function [srcmasstmp] = srcmms_mass( x, y)
%
%Uses global variable(s): rho, rpi, rlength
%Inputs: x, y
%To modify: <none>
%Returns: srcmms_mass
% srcmasstmp; % Define return value for srcmms_mass as % precision

% dudx; 	% Temp variable: u velocity gradient in x direction
% dvdy;  % Temp variable: v velocity gradient in y direction

% This function returns the MMS mass source term

global rho rpi rlength
global phix phiy phixy apx apy apxy

dudx = phix(2)*apx(2)*rpi/rlength*cos(apx(2)*rpi*x/rlength)  ...
    + phixy(2)*apxy(2)*rpi*y/rlength/rlength  ...
    * cos(apxy(2)*rpi*x*y/rlength/rlength);

dvdy = -phiy(3)*apy(3)*rpi/rlength*sin(apy(3)*rpi*y/rlength)  ...
    - phixy(3)*apxy(3)*rpi*x/rlength/rlength  ...
    * sin(apxy(3)*rpi*x*y/rlength/rlength);

srcmasstmp = rho*dudx + rho*dvdy;
end
%************************************************************************
function [srcxmtmtmp] = srcmms_xmtm( x,  y)
%
%Uses global variable(s): rho, rpi, rmu, rlength
%Inputs: x, y
%To modify: <none>
%Returns: srcmms_xmtm

% srcxmtmtmp; % Define return value for srcmms_xmtm as % precision

% dudx; 	% Temp variable: u velocity gradient in x direction
% dudy;  % Temp variable: u velocity gradient in y direction
% termx;        % Temp variable
% termy;        % Temp variable
% termxy;       % Temp variable
% uvel;         % Temp variable: u velocity
% vvel;         % Temp variable: v velocity
% dpdx;         % Temp variable: pressure gradient in x direction
% d2udx2;       % Temp variable: 2nd derivative of u velocity in x direction
% d2udy2;       % Temp variable: 2nd derivative of u velocity in y direction

%This function returns the MMS x-momentum source term

global rho rpi rmu rlength
global phi0 phix phiy phixy apx apy apxy

termx = phix(2)*sin(apx(2)*rpi*x/rlength);
termy = phiy(2)*cos(apy(2)*rpi*y/rlength);
termxy = phixy(2)*sin(apxy(2)*rpi*x*y/rlength/rlength);
uvel = phi0(2) + termx + termy + termxy;

termx = phix(3)*cos(apx(3)*rpi*x/rlength);
termy = phiy(3)*cos(apy(3)*rpi*y/rlength);
termxy = phixy(3)*cos(apxy(3)*rpi*x*y/rlength/rlength);
vvel = phi0(3) + termx + termy + termxy;

dudx = phix(2)*apx(2)*rpi/rlength*cos(apx(2)*rpi*x/rlength) ...
    + phixy(2)*apxy(2)*rpi*y/rlength/rlength  ...
    * cos(apxy(2)*rpi*x*y/rlength/rlength);

dudy = -phiy(2)*apy(2)*rpi/rlength*sin(apy(2)*rpi*y/rlength)  ...
    + phixy(2)*apxy(2)*rpi*x/rlength/rlength  ...
    * cos(apxy(2)*rpi*x*y/rlength/rlength);

dpdx = -phix(1)*apx(1)*rpi/rlength*sin(apx(1)*rpi*x/rlength) ...
    + phixy(1)*apxy(1)*rpi*y/rlength/rlength  ...
    * cos(apxy(1)*rpi*x*y/rlength/rlength);

d2udx2 = -phix(2)*((apx(2)*rpi/rlength).^2)  ...
    * sin(apx(2)*rpi*x/rlength)  ...
    - phixy(2)*((apxy(2)*rpi*y/rlength/rlength).^2)  ...
    * sin(apxy(2)*rpi*x*y/rlength/rlength);

d2udy2 = -phiy(2)*((apy(2)*rpi/rlength).^2)  ...
    * cos(apy(2)*rpi*y/rlength)  ...
    - phixy(2)*((apxy(2)*rpi*x/rlength/rlength).^2)  ...
    * sin(apxy(2)*rpi*x*y/rlength/rlength);

srcxmtmtmp = rho*uvel*dudx + rho*vvel*dudy + dpdx  ...
    - rmu*( d2udx2 + d2udy2 );

end
%************************************************************************
function [srcymtmtmp] = srcmms_ymtm( x, y)
%
%Uses global variable(s): rho, rpi, rmu, rlength
%Inputs: x, y
%To modify: <none>
%Returns: srcmms_ymtm

% srcymtmtmp; % Define return value for srcmms_ymtm as % precision

% dvdx;         % Temp variable: v velocity gradient in x direction
% dvdy;         % Temp variable: v velocity gradient in y direction
% termx;        % Temp variable
% termy;        % Temp variable
% termxy;       % Temp variable
% uvel;         % Temp variable: u velocity
% vvel;         % Temp variable: v velocity
% dpdy;         % Temp variable: pressure gradient in y direction
% d2vdx2;       % Temp variable: 2nd derivative of v velocity in x direction
% d2vdy2;       % Temp variable: 2nd derivative of v velocity in y direction

% This function returns the MMS y-momentum source term

global rho rpi rmu rlength
global phi0 phix phiy phixy apx apy apxy

termx = phix(2)*sin(apx(2)*rpi*x/rlength);
termy = phiy(2)*cos(apy(2)*rpi*y/rlength);
termxy = phixy(2)*sin(apxy(2)*rpi*x*y/rlength/rlength);
uvel = phi0(2) + termx + termy + termxy;

termx = phix(3)*cos(apx(3)*rpi*x/rlength);
termy = phiy(3)*cos(apy(3)*rpi*y/rlength);
termxy = phixy(3)*cos(apxy(3)*rpi*x*y/rlength/rlength);
vvel = phi0(3) + termx + termy + termxy;

dvdx = -phix(3)*apx(3)*rpi/rlength*sin(apx(3)*rpi*x/rlength)  ...
    - phixy(3)*apxy(3)*rpi*y/rlength/rlength  ...
    * sin(apxy(3)*rpi*x*y/rlength/rlength);

dvdy = -phiy(3)*apy(3)*rpi/rlength*sin(apy(3)*rpi*y/rlength)  ...
    - phixy(3)*apxy(3)*rpi*x/rlength/rlength  ...
    * sin(apxy(3)*rpi*x*y/rlength/rlength);

dpdy = phiy(1)*apy(1)*rpi/rlength*cos(apy(1)*rpi*y/rlength)  ...
    + phixy(1)*apxy(1)*rpi*x/rlength/rlength  ...
    * cos(apxy(1)*rpi*x*y/rlength/rlength);

d2vdx2 = -phix(3)*((apx(3)*rpi/rlength).^2)  ...
    * cos(apx(3)*rpi*x/rlength)  ...
    - phixy(3)*((apxy(3)*rpi*y/rlength/rlength).^2)  ...
    * cos(apxy(3)*rpi*x*y/rlength/rlength);

d2vdy2 = -phiy(3)*((apy(3)*rpi/rlength).^2)  ...
    * cos(apy(3)*rpi*y/rlength)  ...
    - phixy(3)*((apxy(3)*rpi*x/rlength/rlength).^2)  ...
    * cos(apxy(3)*rpi*x*y/rlength/rlength);

srcymtmtmp = rho*uvel*dvdx + rho*vvel*dvdy + dpdy  ...
    - rmu*( d2vdx2 + d2vdy2 );

end
%************************************************************************
% function [dtmin] = compute_time_step(dtmin)
function [dtmin] = compute_time_step()
%
%Uses global variable(s): one, two, four, half, fourth
%Uses global variable(s): vel2ref, rmu, rho, dx, dy, cfl, rkappa, imax, jmax
%Uses: u
%To Modify: dt, dtmin

% i                        % i index (x direction)
% j                        % j index (y direction)

% dtvisc       % Viscous time step stability criteria (constant over domain)
% uvel2        % Local velocity squared
% beta2        % Beta squared paramete for time derivative preconditioning
% lambda_x     % Max absolute value eigenvalue in (x,t)
% lambda_y     % Max absolute value eigenvalue in (y,t)
% lambda_max   % Max absolute value eigenvalue (used in convective time step computation)
% dtconv       % Local convective time step restriction

global half 
global vel2ref rmu rho dx dy cfl rkappa imax jmax
global u dt


% !************************************************************** */
% !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
% !************************************************************** */

uvel = u(:,:,2);
vvel = u(:,:,3);
uvel2 = uvel.^2;
vvel2 = vvel.^2;
beta2 = max(uvel2,rkappa*vel2ref);
lambda_x = abs(half*(abs(uvel)+sqrt(uvel2+4*beta2)));
lambda_y = abs(half*(abs(vvel)+sqrt(vvel2+4*beta2)));
lambda_max = max(lambda_x,lambda_y);
dtconv = min(dx,dy)./abs(lambda_max);
dtvisc = dx*dy/(4*rmu/rho);
dt = cfl*min(dtconv,dtvisc); %for Local time stepping
dtmin = min(dt,[],'all');
dt = dtmin*ones(imax,jmax); % for Global time stepping

end
%************************************************************************
function Compute_Artificial_Viscosity(~)
%
%Uses global variable(s): zero, one, two, four, six, half, fourth
%Uses global variable(s): imax, jmax, lim, rho, dx, dy, Cx, Cy, Cx2, Cy2, fsmall, vel2ref, rkappa
%Uses: u
%To Modify: artviscx, artviscy

% i                        % i index (x direction)
% j                        % j index (y direction)

% uvel2        % Local velocity squared
% beta2        % Beta squared paramete for time derivative preconditioning
% lambda_x     % Max absolute value e-value in (x,t)
% lambda_y     % Max absolute value e-value in (y,t)
% d4pdx4       % 4th derivative of pressure w.r.t. x
% d4pdy4       % 4th derivative of pressure w.r.t. y
% % d2pdx2       % 2nd derivative of pressure w.r.t. x [these are not used]
% % d2pdy2       % 2nd derivative of pressure w.r.t. y [these are not used]
% % pfunct1      % Temporary variable for 2nd derivative damping [these are
% not used]
% % pfunct2      % Temporary variable for 2nd derivative damping [these are
% not used]

%global two four six half
global half
global imax jmax  dx dy Cx Cy  vel2ref rkappa %lim rho Cx2 Cy2 fsmall
global u
global artviscx artviscy


% !************************************************************** */
% !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
% !************************************************************** */

uvel = u(:,:,2);
vvel = u(:,:,3);
uvel2 = uvel.^2;
vvel2 = vvel.^2;
beta2 = max(uvel2,rkappa*vel2ref);
lambda_x = abs(half*(abs(uvel)+sqrt(uvel2+4*beta2)));
lambda_y = abs(half*(abs(vvel)+sqrt(vvel2+4*beta2)));
for i = 3:imax-2 %Excluding the Boundaries
   for j = 3:jmax-2
artviscx(i,j) = -((max(lambda_x,[],'all')*Cx*dx^3)./beta2(i,j))*(u(i+2,j,1)-4*u(i+1,j,1)+6*u(i,j,1)-4*u(i-1,j,1)+u(i-2,j,1))/(dx^4);
artviscy(i,j) = -((max(lambda_y,[],'all')*Cy*dy^3)./beta2(i,j))*(u(i,j+2,1)-4*u(i,j+1,1)+6*u(i,j,1)-4*u(i,j-1,1)+u(i,j-2,1))/(dy^4);
   end 
   
end

%Extrapolating Pressure
%Left wall
 i = 2;
 for j = 3:jmax-2
     pi_0 = 2*u(1,j,1)-u(2,j,1);
     artviscx(i,j) = -((max(lambda_x,[],'all')*Cx*dx^3)./beta2(i,j))*(u(i+2,j,1)-4*u(i+1,j,1)+6*u(i,j,1)-4*u(i-1,j,1)+pi_0)/(dx^4);
     artviscy(i,j) = -((max(lambda_y,[],'all')*Cy*dy^3)./beta2(i,j))*(u(i,j+2,1)-4*u(i,j+1,1)+6*u(i,j,1)-4*u(i,j-1,1)+u(i,j-2,1))/(dy^4);
 end
 
 %right wall
  i = imax-1;
 for j = 3:jmax-2
     pi_0 = 2*u(imax,j,1)-u(imax-1,j,1);
     artviscx(i,j) = -((max(lambda_x,[],'all')*Cx*dx^3)./beta2(i,j))*(pi_0-4*u(i+1,j,1)+6*u(i,j,1)-4*u(i-1,j,1)+u(i-2,j,1))/(dx^4);
     artviscy(i,j) = -((max(lambda_y,[],'all')*Cy*dy^3)./beta2(i,j))*(u(i,j+2,1)-4*u(i,j+1,1)+6*u(i,j,1)-4*u(i,j-1,1)+u(i,j-2,1))/(dy^4);
 end

 %bottom wall
 j=2;
for i = 3:imax-2
     pj_0 = 2*u(i,1,1)-u(i,2,1);
     artviscx(i,j) = -((max(lambda_x,[],'all')*Cx*dx^3)./beta2(i,j))*(u(i+2,j,1)-4*u(i+1,j,1)+6*u(i,j,1)-4*u(i-1,j,1)+u(i-2,j,1))/(dx^4);
     artviscy(i,j) = -((max(lambda_y,[],'all')*Cy*dy^3)./beta2(i,j))*(u(i,j+2,1)-4*u(i,j+1,1)+6*u(i,j,1)-4*u(i,j-1,1)+pj_0)/(dy^4);
end
 
%top wall
j=jmax-1;
for i = 3:imax-2
     pj_0 = 2*u(i,jmax,1)-u(i,jmax-1,1);
     artviscx(i,j) = -((max(lambda_x,[],'all')*Cx*dx^3)./beta2(i,j))*(u(i+2,j,1)-4*u(i+1,j,1)+6*u(i,j,1)-4*u(i-1,j,1)+u(i-2,j,1))/(dx^4);
     artviscy(i,j) = -((max(lambda_y,[],'all')*Cy*dy^3)./beta2(i,j))*(pj_0-4*u(i,j+1,1)+6*u(i,j,1)-4*u(i,j-1,1)+u(i,j-2,1))/(dy^4);
end

%Extrapolating Artificial Viscosity
%left wall

% for j = 2:jmax-1
%     i = 2;
%     artviscx(i,j) = 2*artviscx(i+1,j)-artviscx(i+2,j);
%     artviscy(i,j) = 2*artviscy(i+1,j)-artviscy(i+2,j);
% end

%right wall

% for j = 2:jmax-1
%     i = imax-1;
%     artviscx(i,j) = 2*artviscx(i-1,j)-artviscx(i-2,j);
%     artviscy(i,j) = 2*artviscy(i-1,j)-artviscy(i-2,j);
% end

%top wall
% for i = 2:imax-1
%     j = jmax-1;
%     artviscx(i,j) = 2*artviscx(i,j-1)-artviscx(i,j-2);
%     artviscy(i,j) = 2*artviscy(i,j-1)-artviscy(i,j-2);
% end

%bottom wall
% for i = 2:imax-1
%     j = 2;
%     artviscx(i,j) = 2*artviscx(i,j+1)-artviscx(i,j+2);
%     artviscy(i,j) = 2*artviscy(i,j+1)-artviscy(i,j+2);
% end

%extrapolating Artificial viscosity at corners
  % artviscx(2,2) = 1/3*(artviscx(2,3)+artviscx(3,3)+artviscx(3,2)); %bottom left corner
  
%method 2  
  ax = 2*artviscx(3,2)-artviscx(4,2);
  ay = 2*artviscx(2,3)-artviscx(2,4);
  artviscx(2,2) = 0.5*(ax+ay);

%method3
 

 %artviscx(imax-1,2) = 1/3*(artviscx(imax-1,3)+artviscx(imax-2,3)+artviscx(imax-2,2)); %bottom right corner
 
 %method 2
  ax = 2*artviscx(imax-2,2)-artviscx(imax-3,2);
  ay = 2*artviscx(imax-1,3)-artviscx(imax-1,4);
  artviscx(imax-1,2) = 0.5*(ax+ay);
  
 %artviscx(imax-1,jmax-1) = 1/3*(artviscx(imax-2,jmax-1)+artviscx(imax-2,jmax-2)+artviscx(imax-1,jmax-2)); %top right corner
 
  ax = 2*artviscx(imax-2,jmax-1)-artviscx(imax-3,jmax-1);
  ay = 2*artviscx(imax-1,jmax-2)-artviscx(imax-1,jmax-3);
  artviscx(imax-1,jmax-1) = 0.5*(ax+ay);
  
 %artviscx(2,jmax-1) = 1/3*(artviscx(3,jmax-1)+artviscx(3,jmax-2)+artviscx(2,jmax-2)); %top left corner
  ax = 2*artviscx(3,jmax-1)-artviscx(4,jmax-1);
  ay = 2*artviscx(2,jmax-2)-artviscx(2,jmax-3);
  artviscx(2,jmax-1) = 0.5*(ax+ay);

 %artviscy(2,2) = 1/3*(artviscy(2,3)+artviscy(3,3)+artviscy(3,2)); %bottom left corner
 
  ax = 2*artviscy(3,2)-artviscy(4,2);
  ay = 2*artviscy(2,3)-artviscy(2,4);
  artviscy(2,2) = 0.5*(ax+ay);
  
 %artviscy(imax-1,2) = 1/3*(artviscy(imax-1,3)+artviscy(imax-2,3)+artviscy(imax-2,2)); %bottom right corner
 
  ax = 2*artviscy(imax-2,2)-artviscy(imax-3,2);
  ay = 2*artviscy(imax-1,3)-artviscy(imax-1,4);
  artviscy(imax-1,2) = 0.5*(ax+ay);
 
 %artviscy(imax-1,jmax-1) = 1/3*(artviscy(imax-2,jmax-1)+artviscy(imax-2,jmax-2)+artviscy(imax-1,jmax-2)); %top right corner
 
  ax = 2*artviscy(imax-2,jmax-1)-artviscy(imax-3,jmax-1);
  ay = 2*artviscy(imax-1,jmax-2)-artviscy(imax-1,jmax-3);
  artviscy(imax-1,jmax-1) = 0.5*(ax+ay);
  
 %artviscy(2,jmax-1) = 1/3*(artviscy(3,jmax-1)+artviscy(3,jmax-2)+artviscy(2,jmax-2)); %top left corner
 
  ax = 2*artviscy(3,jmax-1)-artviscy(4,jmax-1);
  ay = 2*artviscy(2,jmax-2)-artviscy(2,jmax-3);
  artviscy(2,jmax-1) = 0.5*(ax+ay);
 
end
%************************************************************************
function SGS_forward_sweep(~)
%
%Uses global variable(s): two, three, six, half
%Uses global variable(s): imax, imax, jmax, ipgorder, rho, rhoinv, dx, dy, rkappa, ...
%                      xmax, xmin, ymax, ymin, rmu, vel2ref
%Uses: artviscx, artviscy, dt, s
%To Modify: u

% i                        % i index (x direction)
% j                        % j index (y direction)

% dpdx         % First derivative of pressure w.r.t. x
% dudx         % First derivative of x velocity w.r.t. x
% dvdx         % First derivative of y velocity w.r.t. x
% dpdy         % First derivative of pressure w.r.t. y
% dudy         % First derivative of x velocity w.r.t. y
% dvdy         % First derivative of y velocity w.r.t. y
% d2udx2       % Second derivative of x velocity w.r.t. x
% d2vdx2       % Second derivative of y velocity w.r.t. x
% d2udy2       % Second derivative of x velocity w.r.t. y
% d2vdy2       % Second derivative of y velocity w.r.t. y
% beta2        % Beta squared parameter for time derivative preconditioning
% uvel2        % Velocity squared

% global %two half
global imax jmax rho rhoinv dx dy rkappa rmu vel2ref
global artviscx artviscy dt s u

% Symmetric Gauss-Siedel: Forward Sweep

% !************************************************************** */
% !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
% !************************************************************** */
uvel2 = u(:,:,2).^2+u(:,:,3).^2;
beta2 = max(uvel2,rkappa*vel2ref);  

for i=2:imax-1
    for j=2:jmax-1
        uij = u(i,j,2);
        ui_p1 = u(i+1,j,2);
        ui_m1 = u(i-1,j,2);
        uj_p1 = u(i,j+1,2);
        uj_m1 = u(i,j-1,2);
        vij = u(i,j,3);
        vi_p1 = u(i+1,j,3);
        vi_m1 = u(i-1,j,3);
        vj_p1 = u(i,j+1,3);
        vj_m1 = u(i,j-1,3);
        pij = u(i,j,1);
        pi_p1 = u(i+1,j,1);
        pi_m1 = u(i-1,j,1);
        pj_p1 = u(i,j+1,1);
        pj_m1 = u(i,j-1,1);
        
        u(i,j,1) = pij-beta2(i,j)*dt(i,j).*(rho*(ui_p1-ui_m1)/(2*dx) +rho*(vj_p1-vj_m1)/(2*dy)-artviscx(i,j)-artviscy(i,j)-s(i,j,1));
        u(i,j,2) = uij - dt(i,j)*rhoinv*(rho*uij*(ui_p1-ui_m1)/(2*dx)+rho*vij*(uj_p1-uj_m1)/(2*dy)+(pi_p1-pi_m1)/(2*dx)-rmu*(ui_p1-2*uij+ui_m1)/(dx^2)-rmu*(uj_p1-2*uij+uj_m1)/(dy^2)-s(i,j,2));
        u(i,j,3) = vij - dt(i,j)*rhoinv*(rho*uij*(vi_p1-vi_m1)/(2*dx)+rho*vij*(vj_p1-vj_m1)/(2*dy)+(pj_p1-pj_m1)/(2*dy)-rmu*(vi_p1-2*vij+vi_m1)/(dx^2)-rmu*(vj_p1-2*vij+vj_m1)/(dy^2)-s(i,j,3));
        
    end
end
   


end
%************************************************************************
function SGS_backward_sweep(~)
%
%Uses global variable(s): two, three, six, half
%Uses global variable(s): imax, imax, jmax, ipgorder, rho, rhoinv, dx, dy, rkappa, ...
%                      xmax, xmin, ymax, ymin, rmu, vel2ref
%Uses: artviscx, artviscy, dt, s
%To Modify: u

% i                        % i index (x direction)
% j                        % j index (y direction)

% dpdx         % First derivative of pressure w.r.t. x
% dudx         % First derivative of x velocity w.r.t. x
% dvdx         % First derivative of y velocity w.r.t. x
% dpdy         % First derivative of pressure w.r.t. y
% dudy         % First derivative of x velocity w.r.t. y
% dvdy         % First derivative of y velocity w.r.t. y
% d2udx2       % Second derivative of x velocity w.r.t. x
% d2vdx2       % Second derivative of y velocity w.r.t. x
% d2udy2       % Second derivative of x velocity w.r.t. y
% d2vdy2       % Second derivative of y velocity w.r.t. y
% beta2        % Beta squared parameter for time derivative preconditioning
% uvel2        % Velocity squared

% global two half
global imax jmax rho rhoinv dx dy rkappa rmu vel2ref
global artviscx artviscy dt s u

% Symmetric Gauss-Siedel: Backward Sweep

% !************************************************************** */
% !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
% !************************************************************** */

uvel2 = u(:,:,2).^2+u(:,:,3).^2;
beta2 = max(uvel2,rkappa*vel2ref);  

for i=imax-1:2
    for j=jmax-1:2
        uij = u(i,j,2);
        ui_p1 = u(i+1,j,2);
        ui_m1 = u(i-1,j,2);
        uj_p1 = u(i,j+1,2);
        uj_m1 = u(i,j-1,2);
        vij = u(i,j,3);
        vi_p1 = u(i+1,j,3);
        vi_m1 = u(i-1,j,3);
        vj_p1 = u(i,j+1,3);
        vj_m1 = u(i,j-1,3);
        pij = u(i,j,1);
        pi_p1 = u(i+1,j,1);
        pi_m1 = u(i-1,j,1);
        pj_p1 = u(i,j+1,1);
        pj_m1 = u(i,j-1,1);
        
        u(i,j,1) = pij-beta2(i,j)*dt(i,j).*(rho*(ui_p1-ui_m1)/(2*dx) +rho*(vj_p1-vj_m1)/(2*dy)-artviscx(i,j)-artviscy(i,j)-s(i,j,1));
        u(i,j,2) = uij - dt(i,j)*rhoinv*(rho*uij*(ui_p1-ui_m1)/(2*dx)+rho*vij*(uj_p1-uj_m1)/(2*dy)+(pi_p1-pi_m1)/(2*dx)-rmu*(ui_p1-2*uij+ui_m1)/(dx^2)-rmu*(uj_p1-2*uij+uj_m1)/(dy^2)-s(i,j,2));
        u(i,j,3) = vij - dt(i,j)*rhoinv*(rho*uij*(vi_p1-vi_m1)/(2*dx)+rho*vij*(vj_p1-vj_m1)/(2*dy)+(pj_p1-pj_m1)/(2*dy)-rmu*(vi_p1-2*vij+vi_m1)/(dx^2)-rmu*(vj_p1-2*vij+vj_m1)/(dy^2)-s(i,j,3));
        
    end
end
   


end



%************************************************************************
function point_Jacobi(~)
%
%Uses global variable(s): two, three, six, half
%Uses global variable(s): imax, imax, jmax, ipgorder, rho, rhoinv, dx, dy, rkappa, ...
%                      xmax, xmin, ymax, ymin, rmu, vel2ref
%Uses: uold, artviscx, artviscy, dt, s
%To Modify: u


% i                        % i index (x direction)
% j                        % j index (y direction)

% dpdx         % First derivative of pressure w.r.t. x
% dudx         % First derivative of x velocity w.r.t. x
% dvdx         % First derivative of y velocity w.r.t. x
% dpdy         % First derivative of pressure w.r.t. y
% dudy         % First derivative of x velocity w.r.t. y
% dvdy         % First derivative of y velocity w.r.t. y
% d2udx2       % Second derivative of x velocity w.r.t. x
% d2vdx2       % Second derivative of y velocity w.r.t. x
% d2udy2       % Second derivative of x velocity w.r.t. y
% d2vdy2       % Second derivative of y velocity w.r.t. y
% beta2        % Beta squared parameter for time derivative preconditioning
% uvel2        % Velocity squared
% global two half
global imax jmax rho rhoinv dx dy rkappa rmu vel2ref
global u uold artviscx artviscy dt s

% Point Jacobi method

% !************************************************************** */
% !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
% !************************************************************** */
uvel2 = u(:,:,2).^2+u(:,:,3).^2;
beta2 = max(uvel2,rkappa*vel2ref);  

for i=2:imax-1
    for j=2:jmax-1
        uij = uold(i,j,2);
        ui_p1 = uold(i+1,j,2);
        ui_m1 = uold(i-1,j,2);
        uj_p1 = uold(i,j+1,2);
        uj_m1 = uold(i,j-1,2);
        vij = uold(i,j,3);
        vi_p1 = uold(i+1,j,3);
        vi_m1 = uold(i-1,j,3);
        vj_p1 = uold(i,j+1,3);
        vj_m1 = uold(i,j-1,3);
        pij = uold(i,j,1);
        pi_p1 = uold(i+1,j,1);
        pi_m1 = uold(i-1,j,1);
        pj_p1 = uold(i,j+1,1);
        pj_m1 = uold(i,j-1,1);
        
        u(i,j,1) = pij-beta2(i,j)*dt(i,j).*(rho*(ui_p1-ui_m1)/(2*dx) +rho*(vj_p1-vj_m1)/(2*dy)-artviscx(i,j)-artviscy(i,j)-s(i,j,1));
        u(i,j,2) = uij - dt(i,j)*rhoinv*(rho*uij*(ui_p1-ui_m1)/(2*dx)+rho*vij*(uj_p1-uj_m1)/(2*dy)+(pi_p1-pi_m1)/(2*dx)-rmu*(ui_p1-2*uij+ui_m1)/(dx^2)-rmu*(uj_p1-2*uij+uj_m1)/(dy^2)-s(i,j,2));
        u(i,j,3) = vij - dt(i,j)*rhoinv*(rho*uij*(vi_p1-vi_m1)/(2*dx)+rho*vij*(vj_p1-vj_m1)/(2*dy)+(pj_p1-pj_m1)/(2*dy)-rmu*(vi_p1-2*vij+vi_m1)/(dx^2)-rmu*(vj_p1-2*vij+vj_m1)/(dy^2)-s(i,j,3));
        
    end
end


end
%************************************************************************
function pressure_rescaling(~)
%
%Uses global variable(s): imax, jmax, imms, xmax, xmin, ymax, ymin, rlength, pinf
%To Modify: u

% i                        % i index (x direction)
% j                        % j index (y direction)

% iref                      % i index location of pressure rescaling point
% jref                      % j index location of pressure rescaling point

% x        % Temporary variable for x location
% y        % Temporary variable for y location
% deltap   % delta_pressure for rescaling all values

global imax jmax imms xmax xmin ymax ymin pinf
global u

iref = (imax-1)/2+1;     % Set reference pressure to center of cavity
jref = (jmax-1)/2+1;
if (imms==1)
    x = (xmax - xmin)*(iref-1)/(imax - 1);
    y = (ymax - ymin)*(jref-1)/(jmax - 1);
    deltap = u(iref,jref,1) - umms(x,y,1); % Constant in MMS
else
    deltap = u(iref,jref,1) - pinf; % Reference pressure
end

j=1:jmax;
i=1:imax;
u(i,j,1) = u(i,j,1) - deltap;


end
%************************************************************************
function [res, resinit, conv] = check_iterative_convergence...
    (n, res, resinit, ninit, rtime, dtmin)
%
%Uses global variable(s): zero
%Uses global variable(s): imax, jmax, neq, fsmall
%Uses: n, u, uold, dt, res, resinit, ninit, rtime, dtmin
%To modify: conv

% i                        % i index (x direction)
% j                        % j index (y direction)
% k                        % k index (# of equations)

%global zero neq fsmall
global imax jmax 
global u uold dt fp1

% Compute iterative residuals to monitor iterative convergence

% !************************************************************** */
% !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
% !************************************************************** */
%L2 Norm
DE = (u-uold)./dt;
res(1) = sqrt(sum(DE(:,:,1).^2,'all')/(imax*jmax));
res(2) = sqrt(sum(DE(:,:,2).^2,'all')/(imax*jmax));
res(3) = sqrt(sum(DE(:,:,3).^2,'all')/(imax*jmax));
% if(n==ninit)
%     resinit = res;
% end
res = res./resinit;

conv = max(res);


% Write iterative residuals every 10 iterations
if ( (mod(n,10)==0)||(n==ninit) )
    fprintf(fp1, '%d %e %e %e %e\n',n, rtime, res(1), res(2), res(3) );
    fprintf('%d   %e   %e   %e   %e   %e\n',n, rtime, dtmin, res(1), res(2), res(3) );
    % Maybe a need to format this better
end

% Write header for iterative residuals every 200 iterations
if ( (mod(n,200)==0)||(n==ninit) )
    fprintf('Iter. Time (s)   dt (s)      Continuity    x-Momentum    y-Momentum\n');
end

end

%************************************************************************
function [rL1norm, rL2norm, rLinfnorm] = Discretization_Error_Norms(filename)
%
%Uses global variable(s): zero
%Uses global variable(s): imax, jmax, neq, imms, xmax, xmin, ymax, ymin, rlength
%Uses: u
%To modify: rL1norm, rL2norm, rLinfnorm


% i                        % i index (x direction)
% j                        % j index (y direction)
% k                        % k index (# of equations)

% x        % Temporary variable for x location
% y        % Temporary variable for y location
% DE   	% Discretization error (absolute value)

global imax jmax  imms u ummsArray % neq xmax xmin ymax ymin 

if imms==1
  
% !************************************************************** */
% !************ADD CODING HERE FOR INTRO CFD STUDENTS************ */
% !************************************************************** */
%L2 Norm
DE = abs(u-ummsArray);
rL2norm(1) = sqrt(sum(DE(:,:,1).^2,'all')/(imax*jmax));
rL2norm(2) = sqrt(sum(DE(:,:,2).^2,'all')/(imax*jmax));
rL2norm(3) = sqrt(sum(DE(:,:,3).^2,'all')/(imax*jmax));

%Linf Norm
rLinfnorm(1) = (max(DE(:,:,1),[],'all'));
rLinfnorm(2) = (max(DE(:,:,2),[],'all'));
rLinfnorm(3) = (max(DE(:,:,3),[],'all'));

%L1 Norm
rL1norm(1) = (sum(DE(:,:,1),'all')/(imax*jmax));
rL1norm(2) = (sum(DE(:,:,2),'all')/(imax*jmax));
rL1norm(3) = (sum(DE(:,:,3),'all')/(imax*jmax));

%filename
save([filename,'_errors.mat'],'imax','rL1norm','rL2norm','rLinfnorm');

else
    rL2norm = zeros(1,3);
    rLinfnorm = zeros(1,3);
    rL1norm = zeros(1,3);

end

end
