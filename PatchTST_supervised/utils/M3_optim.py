#FILE_ADDED_BY_ME

import torch
from torch.optim import Optimizer

class M3Optimizer(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.95), beta3=0.9, alpha=0.1, ns_steps=5, eps=1e-8):
        # Añadimos ns_steps (número de iteraciones de Newton-Schulz, por defecto 5 según el paper)
        defaults = dict(lr=lr, betas=betas, beta3=beta3, alpha=alpha, ns_steps=ns_steps, eps=eps)
        super(M3Optimizer, self).__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            ns_steps = group['ns_steps']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                state = self.state[p]

                # 1. INICIALIZACIÓN DEL ESTADO
                if len(state) == 0:
                    state['step'] = 0
                    state['m1'] = torch.zeros_like(p) # Nivel 1 (Memoria Rápida)
                    state['m2'] = torch.zeros_like(p) # Nivel 2 (Memoria Lenta/Anidada)
                    
                    # Solo necesitamos calcular la varianza 'v' para parámetros 1D
                    if p.dim() < 2:
                        state['v'] = torch.zeros_like(p)

                state['step'] += 1
                m1, m2 = state['m1'], state['m2']

                # 2. ACTUALIZACIÓN MULTI-ESCALA (Nested Learning)
                # Nivel rápido (siempre se actualiza)
                m1.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Nivel lento (frecuencia reducida)
                if state['step'] % 10 == 0: 
                    m2.mul_(group['beta3']).add_(grad, alpha=1 - group['beta3'])

                # Combinamos ambas memorias
                combined_grad = m1 + group['alpha'] * m2

                # 3. APLICACIÓN DEL GRADIENTE (Muon vs Adam)
                if p.dim() >= 2:
                    # --- RUTINA MUON (Para matrices de pesos) ---
                    # Aplanamos a 2D por si es un tensor de más dimensiones (ej. Conv2d)
                    orig_shape = combined_grad.shape
                    G = combined_grad.view(orig_shape[0], -1)
                    
                    # Iteraciones de Newton-Schulz para ortogonalizar
                    X = G / (G.norm() + group['eps'])
                    for _ in range(ns_steps):
                        # Fórmula: X = 1.5 * X - 0.5 * X * (X^T * X)
                        A = X.T @ X
                        X = 1.5 * X - 0.5 * (X @ A)
                        
                    orthogonal_grad = X.view(orig_shape)
                    
                    # Actualizamos los pesos con el gradiente ortogonalizado
                    p.add_(orthogonal_grad, alpha=-group['lr'])
                    
                else:
                    # --- RUTINA ADAM (Para sesgos y normalizaciones 1D) ---
                    v = state['v']
                    v.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                    
                    denom = (v.sqrt() / (1 - beta2**state['step'])).add_(group['eps'])
                    p.addcdiv_(combined_grad, denom, value=-group['lr'])