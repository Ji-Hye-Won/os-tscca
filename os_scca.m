function [u,v,w, objFun] = os_scca(data, para)
% --------------------------------------------------------------------
% objective specific sparse canonical correlation
% handle fused lasso and brain connectivity 
% --------------------------------------------------------------------
% Input:
%       - data.X, n*p matrix of neuroimaging data
%       - data.Y, n*q matrix of genetic data
%       - data.Z, n*1 vector of objective-specific term
%       - paras: lambda_u and lambda_v
%
% Output:
%       - u, the canonical loading vector for X
%       - v, the canonical loading vector for X
%       - w, a scalar (it can be ignored)
%---------------------------------------------------------------------
% Author: Ji Hye Won and Mansu Kim (jihyelo100@gmail.com)
% Date created: Apr-21-2019
% Updated: May-09-2020
% @Sungkyunkwan univertity.
% --------------------------------------------------------------------

% Set data
X = data.X;
Y = data.Y;
Z = data.Z;

p = size(X,2);
q = size(Y,2);
r = 1;

% Setting
XX = X'*X;
YY = Y'*Y;
ZZ = Z'*Z;

XY = X'*Y; YX = Y'*X; ZX = Z'*X;
XZ = X'*Z; YZ = Y'*Z; ZY = Z'*Y;

b1 = para(1);
b2 = para(2);

% Initialization
u= ones(p,1)./p;
v = ones(q,1)./q;

d1 = ones(p, 1);
d2 = ones(q, 1);
d3 = 1;

max_iter = 100;
err = 0.01; % 0.01 ~ 0.05
diff_obj = err*10;
i = 0;

while ((i < max_iter) && (diff_obj > err) )
    i = i +1;
    D1 = diag(d1);
    u_new = (XX+b1*D1)\(XY*v+XZ);
    scale_u = sqrt(u_new'*XX*u_new);
    u = u_new./scale_u;
    
%     if sum(isnan(u_new))
%         continue;
%     end
    
    d1 = 0.5 ./(abs(u)+eps);
    
    
    D2 = diag(d2);
    v_new = (YY+b2*D2)\(YX*u+YZ);
    scale_v = sqrt(v_new'*YY*v_new);
    v= v_new./scale_v;
    
%     if sum(isnan(v))
%         continue;
%     end
    
    d2 = 0.5 ./(abs(v)+eps);
           
    D3 = d3;
    w_new = (ZZ+D3)\(ZX*u+ZY*v);
    scale_w = sqrt(w_new'*ZZ*w_new);
    w = w_new./scale_w;
    
%     if sum(isnan(w))
%         continue;
%     end
    
    d3 = 0.5 ./(abs(w)+eps);
  
    % Cost function 
    objFun(i) = -u'*X'*Y*v -v'*Y'*Z*w -w*Z'*X*u + b1*sum(abs(u))+ b2*sum(abs(v));
    
    %Convergence
    if i ~= 1
        diff_obj = abs((objFun(i)-objFun(i-1))/objFun(i-1)); % relative prediction error
    end
    
end

end












