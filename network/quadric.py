import torch
import torch.nn as nn
import torch.nn.functional as F


def grad_check_hook(name, threshold=100.0):
    def hook(grad):
        norm = grad.norm().item()
        if not torch.isnan(grad).any() and norm > threshold:
            print(f"Gradient exploded: {name} grad norm = {norm:.4f}")
            import pdb; pdb.set_trace()
    return hook

class QuadraticSurface(nn.Module):
    def __init__(self):
        super(QuadraticSurface, self).__init__()
    """
    We assume that patches are already decentalized
    """    
    def quadratic_function(self, u, positions):
        is_patch = positions.ndim == 3

        if not is_patch:
            positions = positions.unsqueeze(1)  # → (B, 1, 3)

        x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
        phi = torch.stack([x**2, y**2, z**2, x*y, x*z, y*z, x, y, z], dim=-1) #  (B,N,10)
    
        fval = phi @ u.unsqueeze(-1)  # (B, N, 1)
        fval = fval.squeeze(-1)  # (B, N)
        if not is_patch:
            fval = fval.squeeze(1)  # → (B,)
        
        return fval

    def fit(s, weights, positions, eps=1e-8):
        batch_size, num, _ = positions.shape
        x, y, z = positions[..., 0], positions[..., 1], positions[..., 2]
        phi = torch.stack([x**2, y**2, z**2, x*y, x*z, y*z, x, y, z], dim=-1) # (B,N,9)

        phi_outer = phi.unsqueeze(-1) @ phi.unsqueeze(-2)  # (B,9,9)
       
        weighted_sum = weights.unsqueeze(-1).unsqueeze(-1) * phi_outer   # (B,9,9)
        A = weighted_sum.sum(dim=1)  # (B,9,9)

        B = torch.diag(torch.tensor([0, 0, 0, 0, 0, 0, 1, 1, 1], dtype=torch.float32)).to(A.device)
        B = B.unsqueeze(0).expand(batch_size, -1, -1)

        A00 = A[:, :6, :6]  # Shape: (B, 6, 6)
        A01 = A[:, :6, 6:]  # Shape: (B, 6, 3)
        A10 = A[:, 6:, :6]  # Shape: (B, 3, 6)
        A11 = A[:, 6:, 6:]  # Shape: (B, 3, 3)   
     
        # eye = torch.eye(A00.size(-1), device=A00.device)
        # Tikhonov regulizer
        # A00 = A00 + eps * eye
        # A00_inv = torch.linalg.inv(A00)

        try:
            A00_inv = torch.linalg.inv(A00) 
        except RuntimeError: # usually not happen
            A00_inv = torch.linalg.pinv(A00) 

        A_tilde = A11 - A10 @ A00_inv @ A01 
        A_tilde = (A_tilde + A_tilde.transpose(-1, -2)) / 2.0
        eigenvalues, eigenvectors = torch.linalg.eigh(A_tilde)

        # A.retain_grad()
        # A.register_hook(grad_check_hook("A"))
        # B.retain_grad()
        # B.register_hook(grad_check_hook("B"))

        lambdas = eigenvalues[:, 0] 
        u1 = eigenvectors[:, :, 0] # already normalized 
        u0 = -A00_inv @ A01 @ u1.unsqueeze(2)

        u = torch.cat((u0.squeeze(), u1), dim=1) 
        return u
    
    def get_gradients(self, u, q):
        is_patch = q.ndim == 3  # (B, N, 3) if patch; else (B, 3)

        if not is_patch:
            q = q.unsqueeze(1)  # → (B, 1, 3)
            
        a, b, c = u[:, 0], u[:, 1], u[:, 2]
        d, e, f = u[:, 3], u[:, 4], u[:, 5]
        g, h, i = u[:, 6], u[:, 7], u[:, 8]

        # Unsqueeze to match patch dim (B, 1) → (B, N)
        a, b, c = a[:, None], b[:, None], c[:, None]
        d, e, f = d[:, None], e[:, None], f[:, None]
        g, h, i = g[:, None], h[:, None], i[:, None]

        x, y, z = q[..., 0], q[..., 1], q[..., 2]  # (B, N)

        fx = 2.0 * a * x + d * y + e * z + g
        fy = 2.0 * b * y + d * x + f * z + h
        fz = 2.0 * c * z + e * x + f * y + i

        grads = torch.stack([fx, fy, fz], dim=-1)  # (B, N, 3)

        if not is_patch:
            grads = grads.squeeze(1)  # → (B, 3)
        
        return grads
    
    def get_hessians(self, u, positions):
        is_patch = positions.ndim == 3  # (B, N, 3) or (B, 3)

        if not is_patch:
            positions = positions.unsqueeze(1)  # → (B, 1, 3)

        batch_size, num, _ = positions.shape

        a, b, c = u[:, 0], u[:, 1], u[:, 2]
        d, e, f = u[:, 3], u[:, 4], u[:, 5]

        fxx, fyy, fzz = 2 * a, 2 * b, 2 * c
        fxy, fxz, fyz = d, e, f

        # Allocate
        hessians = torch.zeros(batch_size, num, 3, 3, device=u.device)

        hessians[:, :, 0, 0] = fxx[:, None]
        hessians[:, :, 1, 1] = fyy[:, None]
        hessians[:, :, 2, 2] = fzz[:, None]

        hessians[:, :, 0, 1] = fxy[:, None]
        hessians[:, :, 1, 0] = fxy[:, None]  # Symmetric

        hessians[:, :, 0, 2] = fxz[:, None]
        hessians[:, :, 2, 0] = fxz[:, None] # Symmetric

        hessians[:, :, 1, 2] = fyz[:, None]
        hessians[:, :, 2, 1] = fyz [:, None] # Symmetric

        if not is_patch:
            hessians = hessians.squeeze(1)

        return hessians
    

    def get_origin_normal(self, u):
        n_est = u[:, 6 : 9]
        return n_est
    
    def get_origin_principal_curvatures(self, u):
        batch_size, _ = u.shape
        # first fundemental form
        g, h, i = u[:, 6], u[:, 7], u[:, 8]
        fx, fy, fz = g, h, i
        E = 1 + fx ** 2 / fz ** 2
        F = fx * fy / fz ** 2
        G = 1 + fy ** 2 / fz ** 2
        I = torch.stack([torch.stack([E, F], dim=1),
                torch.stack([F, G], dim=1)], dim=1) 
        
        # second fundemental form
        a, b, c = u[:, 0], u[:, 1], u[:, 2]
        d, e, f = u[:, 3], u[:, 4], u[:, 5]
        fxx, fyy, fzz = 2 * a, 2 * b, 2 * c
        fxy, fxz, fyz = d, e, f
        
        L_Matrix = torch.stack([fxx, fxz, fx,
                        fxz, fzz, fz,
                        fx, fz, torch.zeros_like(fx)], dim=1).reshape(batch_size, 3, 3)
        L = 1.0 / fz**2 * torch.det(L_Matrix)
        
        M_Matrix = torch.stack([fxy, fyz, fy,
                        fxz, fzz, fz,
                        fx, fz, torch.zeros_like(fx)], dim=1).reshape(batch_size, 3, 3)
        M = 1.0 / fz**2 * torch.det(M_Matrix)
        
        N_Matrix = torch.stack([fyy, fyz, fy,
                        fyz, fzz, fz,
                        fy, fz, torch.zeros_like(fx)], dim=1).reshape(batch_size, 3, 3)
        N = 1.0 / fz**2 * torch.det(N_Matrix)

        II = torch.stack([torch.stack([L, M], dim=1),
                torch.stack([M, N], dim=1)], dim=1) 

        W = torch.linalg.inv(I) @ II # I is always invertible at (0,0,0)
        W = (W + W.transpose(-1, -2)) / 2.0
       
        curvatures = torch.linalg.eigvalsh(W)

        return curvatures, _

    def get_origin_gaussian_mean_curvatures(self, u):
        grad = u[:, 6 : 9] # grad norm is always 1 at origin

        a, b, c = u[:, 0], u[:, 1], u[:, 2]
        d, e, f = u[:, 3], u[:, 4], u[:, 5]

        fxx, fxy, fxz = 2 * a, d, e
        fyx, fyy, fyz = fxy, 2 * b, f
        fzx, fzy, fzz = fxz, fyz, 2 * c

        hess = torch.stack([
                torch.stack([fxx, fxy, fxz], dim=-1),
                torch.stack([fyx, fyy, fyz], dim=-1),
                torch.stack([fzx, fzy, fzz], dim=-1)
                ], dim=-2)  # (B, 3, 3)


        # Adjugate matrix of Hessian
        # compute per entry of adj(H) manually
        adj_hess = torch.empty_like(hess)

        adj_hess[..., 0, 0] = fyy * fzz - fyz * fzy
        adj_hess[..., 0, 1] = fyz * fzx - fyx * fzz
        adj_hess[..., 0, 2] = fyx * fzy - fyy * fzx

        adj_hess[..., 1, 0] = fxz * fzy - fxy * fzz
        adj_hess[..., 1, 1] = fxx * fzz - fxz * fzx
        adj_hess[..., 1, 2] = fxy * fzx - fxx * fzy

        adj_hess[..., 2, 0] = fxy * fyz - fxz * fyy
        adj_hess[..., 2, 1] = fyx * fxz - fxx * fyz
        adj_hess[..., 2, 2] = fxx * fyy - fxy * fyx

        # Gaussian curvature
        # numerator: grad^T * adj(hess) * grad
        Kg = torch.einsum('bi,bij,bj->b', grad, adj_hess, grad)  # (B,) 

        # Mean curvature
        gHg = torch.einsum('bi,bij,bj->b', grad, hess, grad)     # grad^T * adj(hess) * grad, shape (B, N, 3)
        trace_hess = torch.sum(torch.diagonal(hess, dim1=-2, dim2=-1), dim=-1)
        Km = (gHg - trace_hess) / 2.0
       
        return Kg, Km

    def get_gaussian_mean_curvatures(self, u, q):
        is_patch = q.ndim == 3  # (B, N, 3) or (B, 3)

        if not is_patch:
            q = q.unsqueeze(1)  # → (B, 1, 3)

        grad = self.get_gradients(u, q)
        hess = self.get_hessians(u, q)
 
        fxx, fxy, fxz = hess[..., 0, 0], hess[..., 0, 1], hess[..., 0, 2]
        fyx, fyy, fyz = hess[..., 1, 0], hess[..., 1, 1], hess[..., 1, 2]
        fzx, fzy, fzz = hess[..., 2, 0], hess[..., 2, 1], hess[..., 2, 2]

        # Adjugate matrix of Hessian
        adj_hess = torch.empty_like(hess)
        adj_hess[..., 0, 0] = fyy * fzz - fyz * fzy
        adj_hess[..., 0, 1] = fyz * fzx - fyx * fzz
        adj_hess[..., 0, 2] = fyx * fzy - fyy * fzx

        adj_hess[..., 1, 0] = fxz * fzy - fxy * fzz
        adj_hess[..., 1, 1] = fxx * fzz - fxz * fzx
        adj_hess[..., 1, 2] = fxy * fzx - fxx * fzy

        adj_hess[..., 2, 0] = fxy * fyz - fxz * fyy
        adj_hess[..., 2, 1] = fyx * fxz - fxx * fyz
        adj_hess[..., 2, 2] = fxx * fyy - fxy * fyx

        grad_sq = torch.sum(grad * grad, dim=-1)
        # Gaussian curvature
        # numerator: grad^T * adj(hess) * grad
        Kg_num = torch.einsum('bni,bnij,bnj->bn', grad, adj_hess, grad)  # (B,N) 
        Kg = Kg_num / grad_sq ** 2

        # Mean curvature
        gHg = torch.einsum('bni,bnij,bnj->bn', grad, hess, grad)     # grad^T * adj(hess) * grad, shape (B, N, 3)
        trace_hess = torch.sum(torch.diagonal(hess, dim1=-2, dim2=-1), dim=-1)
        Km = (gHg - grad_sq * trace_hess) / (2.0 * torch.sqrt(grad_sq) ** 3) 

        if not is_patch:
            Kg = Kg.squeeze(1)  
            Km = Km.squeeze(1) 
       
        return Kg, Km

    def get_principal_curvatures_from_gaussian_mean(self, Kg, Km):
        # Compute discriminant
        discriminant = Km**2 - Kg

        # Ensure numerical stability: clamp negative values to 0
        discriminant = torch.clamp(discriminant, min=0.0)

        root = torch.sqrt(discriminant)

        # Compute principal curvatures
        k1 = Km + root
        k2 = Km - root

        return k1, k2
    
    def get_weingarten(self, u, q, eps=1e-10):
        is_patch = q.ndim == 3
        if not is_patch:
            q = q.unsqueeze(1)  # → (B, 1, 3)

        batch_size, num, _ = q.shape
        
        a, b, c = u[:, 0], u[:, 1], u[:, 2]
        d, e, f = u[:, 3], u[:, 4], u[:, 5]
        g, h, i = u[:, 6], u[:, 7], u[:, 8]

        # Broadcast (B,) → (B, N)
        a, b, c = a[:, None].expand(-1, num), b[:, None].expand(-1, num), c[:, None].expand(-1, num)
        d, e, f = d[:, None].expand(-1, num), e[:, None].expand(-1, num), f[:, None].expand(-1, num)
        g, h, i = g[:, None].expand(-1, num), h[:, None].expand(-1, num), i[:, None].expand(-1, num)
        

        x, y, z = q[..., 0], q[..., 1], q[..., 2]

        # first fundemental form
        fx = 2.0 * a * x + d * y + e * z + g
        fy = 2.0 * b * y + d * x + f * z + h
        fz = 2.0 * c * z + e * x + f * y + i 
        #fz = torch.clamp(fz, min=eps)

        E = 1 + fx ** 2 / fz ** 2
        F = fx * fy / fz ** 2
        G = 1 + fy ** 2 / fz ** 2
        I = torch.stack([torch.stack([E, F], dim=2),
                    torch.stack([F, G], dim=2)], dim=2)
        
        # second fundemental form
        fxx, fxy, fxz = 2 * a, d, e
        fyx, fyy, fyz = fxy, 2 * b, f
        fzx, fzy, fzz = fxz, fyz, 2 * c
        grad = torch.stack([fx, fy, fz], dim=2)
        grad_norm = torch.linalg.norm(grad, dim=2, ord=2) 
        #grad_norm = torch.clamp(grad_norm, min=eps)
        
        L_Matrix = torch.stack([fxx, fxz, fx,
                        fzx, fzz, fz,
                        fx, fz, torch.zeros_like(fx)], dim=-1).reshape(batch_size, num, 3, 3)
        L = 1.0 / fz**2 / grad_norm * torch.det(L_Matrix)
        
        M_Matrix = torch.stack([fxy, fyz, fy,
                        fzx, fzz, fz,
                        fx, fz, torch.zeros_like(fx)], dim=-1).reshape(batch_size, num, 3, 3)
        M = 1.0 / fz**2 / grad_norm * torch.det(M_Matrix)
        
        N_Matrix = torch.stack([fyy, fyz, fy,
                        fzy, fzz, fz,
                        fy, fz, torch.zeros_like(fx)], dim=-1).reshape(batch_size, num, 3, 3)
        N = 1.0 / fz**2 / grad_norm * torch.det(N_Matrix)

        II = torch.stack([torch.stack([L, M], dim=2),
                torch.stack([M, N], dim=2)], dim=2) 

        W = torch.linalg.inv(I) @ II  # (B, N, 2, 2)

        if not is_patch:
            W = W.squeeze(1)  # → (B, 2, 2)

        return W
    
    def get_principal_curvatures_weingarten(self, u, q):
      
        W = self.get_weingarten(u, q) # (B, 2, 2) or (B, N, 2, 2)
        W = (W + W.transpose(-1, -2)) / 2.0  
        curvatures = torch.linalg.eigvalsh(W)

        return curvatures
    
    def get_approximate_projection_order1(self, u, q, eps=1e-10):
        is_patch = q.ndim == 3  # (B, 3) or (B, N, 3) 

        if not is_patch:
            q = q.unsqueeze(1)  # → (B, 1, 3)
        
        grad_f = self.get_gradients(u, q)       # (B, N, 3)
        f = self.quadratic_function(u, q)       # (B, N)

        grad_f_norm_sq = torch.sum(grad_f ** 2, dim=-1)   # (B, N)
        grad_f_norm_sq = torch.clamp(grad_f_norm_sq, min=eps)

        delta1 = (f / grad_f_norm_sq).unsqueeze(-1) * grad_f  # (B, N, 3)

        proj = q - delta1  # (B, N, 3)

        if not is_patch:
            proj = proj.squeeze(1)  # → (B, 3)

        return proj
    
    def get_normal(self, u, q):
        n_est = self.get_gradients(u, q)
        n_est = F.normalize(n_est, dim=-1)

        return n_est
    
    def get_transformed_monge(self, curvatures, directions, normals, eps=1e-10):
        """
        Args:
            curvatures: (B, 2) → [k1, k2]
            directions: (B, 6) → [d1, d2] (principal directions after pca transformation)
            normals:    (B, 3) → normal vectors after pca transformation
        Returns:
            A_pca: (B, 3, 3) quadric matrix in PCA frame
            b_pca: (B, 3)    linear term in PCA frame
        """
        k1, k2 = curvatures[:, 0], curvatures[:, 1]
        d1 = directions[:, 0:3]  # (B, 3)
        d2 = directions[:, 3:6]  # (B, 3)
        n = normals               # (B, 3)

        B = k1.shape[0]

        # Build batched A_monge: (B, 3, 3)
        A_monge = torch.zeros((B, 3, 3), device=k1.device, dtype=k1.dtype)
        A_monge[:, 0, 0] = 0.5 * k1
        A_monge[:, 1, 1] = 0.5 * k2

        # b_monge: (B, 3)
        b_monge = torch.zeros((B, 3), device=k1.device, dtype=k1.dtype)
        b_monge[:, 2] = -1.0  # unit z coefficient in Monge

        # Build R: (B, 3, 3)
        R = torch.stack([d1, d2, n], dim=2)  # (B, 3, 3)

        # Transform quadric
        A_pca = R @ A_monge @ R.transpose(1, 2)  # (B, 3, 3)
        b_pca = torch.matmul(R, b_monge.unsqueeze(-1)).squeeze(-1)  # (B, 3)

        a = A_pca[:, 0, 0]
        b = A_pca[:, 1, 1]
        c = A_pca[:, 2, 2]
        d = A_pca[:, 0, 1] + A_pca[:, 1, 0]
        e = A_pca[:, 0, 2] + A_pca[:, 2, 0]
        f = A_pca[:, 1, 2] + A_pca[:, 2, 1]
        g = b_pca[:, 0]
        h = b_pca[:, 1]
        i = b_pca[:, 2]

        # as b_monge is a unit vector, [g, h, i] is also a unit vector
        u = torch.stack([a, b, c, d, e, f, g, h, i], dim=1)  # (B, 9)

        return u       
 
