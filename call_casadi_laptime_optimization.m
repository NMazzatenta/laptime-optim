function out = call_casadi_laptime_optimization(data)
% Release: 1.0.0, August 2024
% Author: Niccolo' Mazzatenta (contact at nmazzatenta@icloud.com)
%
% See laptimeoptimization.pdf at https://github.com/NMazzatenta/laptime-opti/ for further details

%% CASADI SETUP
import casadi.*

steps = numel(data.v_lim);

% Multiple-shooting setup
% Definition of vector variables and vector function
% sysdynmodel is defined below
v_vec = SX.sym('v_k',steps-1,1);
F_trac_vec = SX.sym('F_trac_k',steps-1,1);
F_brk_vec = SX.sym('F_brk_k',steps-1,1);
slope_vec = SX.sym('slope_k',steps-1,1);
comp_vec = SX.sym('comp_k',steps-1,1);

sysdyn = sysdynmodel(v_vec,F_trac_vec,F_brk_vec,slope_vec, comp_vec);

% Single-shooting setup
% Definition of scalar variables and scalar function
v_k = SX.sym('v_k');
F_trac_k = SX.sym('F_trac_k');
F_brk_k = SX.sym('F_brk_k');
slope_k = SX.sym('slope_k');
comp_k = SX.sym('comp_k');

sysdyn_k = sysdynmodel(v_k,F_trac_k,F_brk_k,slope_k,comp_k);

% Model error (i.e., w) is evaluated from multiple-shooting error
vel = sysdyn('v',data.v_lim(1:end-1),'f_trac',data.f_trac_lim,'f_brk',data.f_brk_lim,'slope',data.road_slope,'comp',DM.zeros(steps-1,1));
out.v_compensation = data.v_lim(2:end) - full(vel.v_p);

%% MAIN RUN TIME

%% IC from System Dynamic Unroll
% choose a first-trial command vector and unroll the dynamics. This step is not mandatory, but may improve convergence time as x_0 already
% satisfies all system dynamics constraints

v_0 = sysdynunroll(data.v_0,data.f_trac_0,data.f_brk_0,out.v_compensation);
out.f_trac_k0 = data.f_trac_0;
out.f_brk_k0 = data.f_brk_0;
out.v_0 = v_0;

%% NLP setup
% Variables
v = SX.sym('v', steps,1);
F_trac = SX.sym('f_trac',steps-1,1);
F_brk = SX.sym('f_brk',steps-1,1);

X   = [ v             ; F_trac                         ; F_brk                         ];
% IC and bounds
X0  = [ v_0           ; data.f_trac_0                  ; data.f_brk_0                  ];
lbw = [ zeros(steps,1); zeros(steps-1,1)               ; data.f_brk_max*ones(steps-1,1)];
ubw = [ data.v_lim    ; data.f_trac_max*ones(steps-1,1); zeros(steps-1,1)              ];

% Ideal energy for traction. Trivial to account for power unit efficiency
E = sum(F_trac.*data.ds);

% System dynamic evaluation
sysdyn_kp1 = sysdyn('v', v(1:end-1),'f_trac',F_trac, 'f_brk', F_brk,'slope',data.road_slope,'comp',out.v_compensation);
v_p = sysdyn_kp1.v_p;

% NL Constraints (system dynamic; closed lap; max power; non concurrent inputs; maximum spendable energy)
G   = [v_p-v(2:end)          ; v(1)-v(end); F_trac.*v(1:end-1)               ; F_trac.*F_brk   ; E                 ];
lbg = [zeros(steps-1,1)      ; 0          ; zeros(steps-1,1)                 ; zeros(steps-1,1); 0                 ];
ubg = [zeros(steps-1,1)      ; 0          ; data.pwr_trac_max*ones(steps-1,1); zeros(steps-1,1); data.enrg_cons_max];

% Objective
J = sum(data.ds/v);

%% NLP definition
opts.ipopt.print_level=3;
opts.print_time=1;

prob = struct('f', J, 'x', X, 'g', G);
solver = nlpsol('solver', 'ipopt', prob, opts);

% Solving and retriving solution
sol = solver('x0', X0, 'lbx', lbw, 'ubx', ubw, 'lbg', lbg, 'ubg', ubg);

% Save state output
x_opt = full(sol.x);
out.v_optim = x_opt(1:steps);
out.f_trac_optim = x_opt(steps+1:2*steps-1);
out.f_brk_optim = x_opt(2*steps:end);

% Save gradients
nlp_grad_f = solver.get_function('nlp_grad_f');
[~,grad_0] = nlp_grad_f(X0,[]);
grad_0 = full(grad_0);
out.grad_v_0 = grad_0(1:steps);
out.grad_f_trac_0 = grad_0(steps+1:2*steps-1);
out.grad_f_brk_0 = grad_0(2*steps:end);

[~,grad_optim] = nlp_grad_f(sol.x,[]);
grad_optim = full(grad_optim);
out.grad_v_optim = grad_optim(1:steps);
out.grad_f_trac_optim = grad_optim(steps+1:2*steps-1);
out.grad_f_brk_optim = grad_optim(2*steps:end);

% Computing additional outputs
out.lap_time = cumtrapz(data.ds,1./out.v_optim); % cumulated lap time
out.cumdlap_time = out.lap_time-data.lap_time; % cumulated delta lap time
out.dlap_time = diff(out.cumdlap_time); % local delta lap time
out.E_local = out.f_trac_optim.*data.ds; % local E traction
out.dE_local = out.E_local-data.f_trac_lim*data.ds; % local delta E traction
out.cumE =cumtrapz(out.E_local); % cumulated E traction
out.cumdE =cumtrapz(out.dE_local); % cumulated delta E traction

%% Sector outputs
% calculate start and stop of power cuts present in optim and not in
% reference
cuts_start = (data.v_lim-out.v_optim)>0.01;
cuts_start_diff = diff(cuts_start);
idxs_cut_start = find(~(cuts_start_diff-1))+1;
cuts_stop = (data.v_lim-out.v_optim)>0.01;
cuts_stop_diff = diff(cuts_stop);
idxs_cut_stop = find(~(cuts_stop_diff+1));

% handle initial/final cuts
if idxs_cut_start(1)>idxs_cut_stop(1)
    idxs_cut_start = [1; idxs_cut_start];
end
if idxs_cut_start(end)>idxs_cut_stop(end)
    idxs_cut_stop = [idxs_cut_stop; numel(data.f_trac_lim)];
end

out.idxs_cut_start = idxs_cut_start;
out.idxs_cut_stop = idxs_cut_stop;
out.path_sector = zeros(numel(idxs_cut_start)*2,1);
out.cumdlap_time_sector = zeros(numel(idxs_cut_start),1);
out.cumdE_sector = zeros(numel(idxs_cut_start),1);

% calculate dt and dE in each cut
for i=1:numel(idxs_cut_start)
    if idxs_cut_stop(i)-idxs_cut_start(i)>2
    mask = data.path_s>=data.path_s(idxs_cut_start(i)) & data.path_s<=data.path_s(idxs_cut_stop(i));
    out.cumdlap_time_sector(i) = trapz(out.dlap_time(mask));
    out.cumdE_sector(i) = trapz(out.dE_local(mask));
    out.path_sector(2*i-1:2*i,1) = [data.path_s(idxs_cut_start(i)); data.path_s(idxs_cut_stop(i))];
    end
end


%% NESTED FUNCTIONS
% System dynamics
    function vdyn_func = sysdynmodel(v_k,F_trac_k,F_brk_k,slope_k,w_k)
        f_drag = data.f0+data.f1.*v_k+data.f2.*v_k.^2 + data.m*data.g.*sin(slope_k);
        dv = (F_trac_k + F_brk_k - f_drag)./data.m.*data.ds./abs(v_k);
        vkp1 = v_k+dv+w_k;
        vdyn_func =Function('VehDyn',{v_k, F_trac_k, F_brk_k,slope_k,w_k},{vkp1},{'v','f_trac','f_brk','slope','comp'},{'v_p'});
    end

% System Dynamics unroll from IC
    function v_unroll = sysdynunroll(v_0,f_trac_k,f_brk_k,w_k)
        v_unroll = DM.zeros(steps,1);
        v_unroll(1) = v_0;
        for step=1:steps-1
            v_kp1 =sysdyn_k('v',v_0,'f_trac',f_trac_k(step),'f_brk',f_brk_k(step),'slope',data.road_slope(step),'comp',w_k(step));
            v_unroll(step+1) = v_kp1.v_p;
            v_0 = v_kp1.v_p;
        end
        v_unroll = full(v_unroll);
    end
end