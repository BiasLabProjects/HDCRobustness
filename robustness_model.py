import torch
import tqdm
import random


class FHRREncoder(object):
    def __init__(self, in_dim, out_dim, kernel="rbf", M=None, device='cpu'):
        
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.kernel = kernel
        self.device = device
        
        if M is None:
            if kernel == 'rbf':
                self.M = torch.empty(out_dim, in_dim, device=device).normal_(mean=0, std=1)
            elif kernel == 'sinc':
                self.M = torch.empty(out_dim, in_dim, device=device).uniform_()
        else:
            self.M = M
    
    def detach(self):
        detached = FHRREncoder(self.in_dim, self.out_dim, self.kernel, M=self.M.detach(), device=self.device)
        return detached
        
    def encode(self, x):
        return torch.exp(1j * torch.tensordot(self.M, x, dims=[[1], [1]])).T

    def __call__(self, x):
        return self.encode(x)
    
    def inner(self, x, y):
        z = torch.tensordot(x, torch.conj(y), dims=[[1], [1]])
        return z.real
    

class ClassificationModel(object):
    
    def __init__(self, encoder, in_dim=2, dim=4000, device='cpu') -> None:
        
        self.encoder = encoder
        self.in_dim = in_dim
        self.dim = dim
        self.device = device
        self.local_points = 10000

        self.class_hvs1 = torch.zeros(dim, dtype=torch.cfloat, device=device)
        self.class_hvs2 = torch.zeros(dim, dtype=torch.cfloat, device=device)

    def detach(self):

        detached = ClassificationModel(self.encoder.detach(), self.in_dim, self.dim, device=self.device)
        detached.class_hvs1 = self.class_hvs1.detach()
        detached.class_hvs2 = self.class_hvs2.detach()
        return detached

    def bundling(self, X, y):

        encoded = self.encoder(X)
        self.class_hvs1 = encoded[y == 0].sum(axis=0)
        self.class_hvs2 = encoded[y == 1].sum(axis=0)
    
    def predict(self, X):

        encoded = self.encoder(X)
        preds = []
        scores = self.encoder.inner(encoded, torch.stack([self.class_hvs1, self.class_hvs2]))
        for i in range(len(X)):
            pred = torch.argmax(scores[i])
            preds.append(pred.item())
        return torch.tensor(preds)

    def get_linear_approximation(self, q):
        '''
        Method 1
        '''

        phi_c1 = self.class_hvs1
        phi_c2 = self.class_hvs2

        v = (phi_c1 - phi_c2)

        Mx = torch.tensordot(self.encoder.M, q.reshape(1, -1), dims=[[1], [1]]).T
        z = -torch.sin(Mx) * v.real + torch.cos(Mx) * v.imag
        sims = torch.linalg.norm(torch.tensordot(self.encoder.M.T, z, dims=[[1], [1]]).T, axis=1)

        L = torch.abs(sims).max()

        return L


    def get_linear_approximation_autograd(self, q):
        '''
        Method 1 with AutoGrad
        '''

        q = q.reshape(1, -1).clone()
        q.requires_grad_(True)
        q.retain_grad()

        phi_c1 = self.class_hvs1
        phi_c2 = self.class_hvs2

        v = (phi_c1 - phi_c2).reshape(1, -1)
        v.requires_grad_(True)

        r = self.encoder.inner(v, self.encoder(q))
        torch.sum(r).backward(retain_graph=True)

        gradients = torch.linalg.norm(q.grad, axis=1)

        L = torch.abs(gradients[0])

        return L

    
    def get_conservative_lipschitz_constant(self, q, r=2):
        '''
        Method 2
        '''

        N = int(self.local_points**(1/self.in_dim))

        if N > 1:
            grids = torch.meshgrid(*[torch.linspace(-r, r, N) for _ in range(self.in_dim)], indexing='ij')
            local_datapoints = torch.stack([grid.reshape(-1) for grid in grids], axis=1)
        else:
            local_datapoints = []
            for _ in range(self.local_points):
                local_datapoints.append(torch.empty(self.in_dim).uniform_(-r, r))
            local_datapoints = torch.stack(local_datapoints)
        local_datapoints += q

        encoded = self.encoder(local_datapoints)
        phi_0 = self.encoder(torch.zeros(1, self.in_dim, dtype=torch.float))

        betas = torch.sqrt(2 * (self.dim - self.encoder.inner(phi_0, encoded))) / torch.linalg.norm(local_datapoints, axis=1)
        beta = betas.max()

        phi_c1 = self.class_hvs1
        phi_c2 = self.class_hvs2
        phi_diff = (phi_c1 - phi_c2).reshape(1, -1)
        L = torch.sqrt(self.encoder.inner(phi_diff, phi_diff)[0, 0]) * beta

        return L


    def get_lipschitz_constant(self):
        '''
        Method 3
        '''

        N = int(self.local_points**(1/self.in_dim))

        phi_c1 = self.class_hvs1
        phi_c2 = self.class_hvs2

        if N > 1:
            grids = torch.meshgrid(*[torch.linspace(-5, 5, N) for _ in range(self.in_dim)], indexing='ij')
            local_datapoints = torch.stack([grid.reshape(-1) for grid in grids], axis=1)
        else:
            local_datapoints = []
            for _ in range(self.local_points):
                local_datapoints.append(torch.empty(self.in_dim).uniform_(-5, 5))
            local_datapoints = torch.stack(local_datapoints)

        v = (phi_c1 - phi_c2)

        Mx = torch.tensordot(self.encoder.M, local_datapoints, dims=[[1], [1]]).T
        z = -torch.sin(Mx) * v.real + torch.cos(Mx) * v.imag
        sims = torch.linalg.norm(torch.tensordot(self.encoder.M.T, z, dims=[[1], [1]]).T, axis=1)

        L = torch.abs(sims).max()

        return L

    def get_lipschitz_constant_autograd(self):
        '''
        Method 3 with AutoGrad
        '''

        N = int(self.local_points**(1/self.in_dim))

        phi_c1 = self.class_hvs1
        phi_c2 = self.class_hvs2

        if N > 1:
            grids = torch.meshgrid(*[torch.linspace(-5, 5, N) for _ in range(self.in_dim)], indexing='ij')
            local_datapoints = torch.stack([grid.reshape(-1) for grid in grids], axis=1)
        else:
            local_datapoints = []
            for _ in range(self.local_points):
                local_datapoints.append(torch.empty(self.in_dim).uniform_(-5, 5))
            local_datapoints = torch.stack(local_datapoints)
        local_datapoints.requires_grad_(True)
        local_datapoints.retain_grad()

        v = (phi_c1 - phi_c2).reshape(1, -1)
        v.requires_grad_(True)

        r = self.encoder.inner(v, self.encoder(local_datapoints))
        torch.sum(r).backward(retain_graph=True)

        gradients = torch.linalg.norm(local_datapoints.grad, axis=1)

        L = torch.abs(gradients).max()

        return L

    def get_lipschitz_constant_gd(self, epochs=10000, lr=0.001, show_pbar=True):
        '''
        Method 2: Global lipschitz constant

        K = max_x ||\nabla r(x)|| 
        '''

        phi_c1 = self.class_hvs1
        phi_c2 = self.class_hvs2

        X = torch.nn.parameter.Parameter(torch.randn(1, self.in_dim))

        optimizer = torch.optim.Adam([X], lr=lr)

        v = (phi_c1 - phi_c2).reshape(1, -1)
        v.requires_grad_(True)

        loss_history = []

        for epoch in tqdm.tqdm(range(epochs), disable=(not show_pbar)):

            optimizer.zero_grad()

            r = self.encoder.inner(v, self.encoder(X))
            # torch.sum(r).backward(retain_graph=True)
            gradient = torch.autograd.grad(torch.sum(r), X, create_graph=True)[0]
            #print(X.grad)
            #gradient = torch.norm(gradient)
            loss = -torch.norm(gradient)
            loss.backward()

            optimizer.step()

            #print(epoch, loss.item())

            loss_history.append(-loss.item())
        
        X = X.detach()

        L = max(loss_history)

        return X, L.item(), loss_history

    def get_conservative_lipschitz_constant_gd(self, epochs=10000, lr=0.001, show_pbar=True):
        '''
        Method 3: Conservative lipschitz constant
        '''

        K_square = self.dim
        phi_0 = self.encoder(torch.zeros(1, self.in_dim, dtype=torch.float))
        phi_0.requires_grad_(True)

        X = torch.nn.parameter.Parameter(torch.randn(1, self.in_dim))

        optimizer = torch.optim.Adam([X], lr=lr)

        loss_history = []

        for epoch in tqdm.tqdm(range(epochs), disable=(not show_pbar)):

            optimizer.zero_grad()

            alpha = -torch.sqrt(2 * (K_square - self.encoder.inner(self.encoder(X), phi_0))) / torch.norm(X)
            alpha.backward()

            optimizer.step()

            #print(epoch, loss.item())

            loss_history.append(-alpha.item())
        
        X = X.detach()

        phi_c1 = self.class_hvs1
        phi_c2 = self.class_hvs2
        phi_diff = (phi_c1 - phi_c2).reshape(1, -1)
        L = torch.sqrt(self.encoder.inner(phi_diff, phi_diff)[0, 0]) * max(loss_history)# [-1]

        return X, L.item(), loss_history

    
    def get_eps(self, q, L):

        phi_diff = self.class_hvs1 - self.class_hvs2
        phi_q = self.encoder(q.reshape(1, -1))
        r_q = self.encoder.inner(phi_diff.reshape(1, -1), phi_q)[0, 0]
        return torch.abs(r_q) / L


class MultiClassificationModel:
    
    def __init__(self, encoder, num_classes, in_dim=2, dim=4000, device='cpu') -> None:
        
        self.encoder = encoder
        self.num_classes = num_classes
        self.in_dim = in_dim
        self.dim = dim
        self.device = device

        self.class_hvs = torch.zeros(num_classes, dim, dtype=torch.cfloat, device=device)
    
    def bundling(self, X, y):

        encoded = self.encoder(X)
        for c in range(self.num_classes):
            self.class_hvs[c] = encoded[y == c].sum(axis=0)
        
        self.models = []
        with tqdm.tqdm(total=self.num_classes * (self.num_classes - 1) // 2) as pbar:
            for c1 in range(self.num_classes):
                for c2 in range(c1 + 1, self.num_classes):
                    model = ClassificationModel(self.encoder, in_dim=self.in_dim, dim=self.dim)
                    binary = torch.zeros_like(y[(y == c1) | (y == c2)])
                    binary[y[(y == c1) | (y == c2)] == c2] = 1
                    model.bundling(X[(y == c1) | (y == c2)], binary)
                    self.models.append(model)

                    pbar.update(1)
    
    def predict(self, X):

        encoded = self.encoder(X)
        preds = []
        scores = self.encoder.inner(encoded, self.class_hvs)
        for i in range(len(X)):
            pred = torch.argmax(scores[i])
            preds.append(pred.item())
        return torch.tensor(preds)

    def get_linear_approximation(self, c1, c2, q):
        '''
        Method 1
        '''

        if c1 > c2: c1, c2 = c2, c1
        i = 0
        for c1_ in range(c1 + 1):
            for c2_ in range(c1 + 1, self.num_classes):
                if c1_ == c1 and c2_ == c2: break
                i += 1
        
        L = self.models[i].get_linear_approximation(q)

        return L


    def get_linear_approximation_autograd(self, c1, c2, q):
        '''
        Method 1 with AutoGrad
        '''

        if c1 > c2: c1, c2 = c2, c1
        i = 0
        for c1_ in range(c1 + 1):
            for c2_ in range(c1 + 1, self.num_classes):
                if c1_ == c1 and c2_ == c2: break
                i += 1
        
        L = self.models[i].get_linear_approximation_autograd(q)

        return L

    
    def get_conservative_lipschitz_constant(self, c1, c2, q, r=2):
        '''
        Method 2
        '''

        if c1 > c2: c1, c2 = c2, c1
        i = 0
        for c1_ in range(c1 + 1):
            for c2_ in range(c1 + 1, self.num_classes):
                if c1_ == c1 and c2_ == c2: break
                i += 1
        
        L = self.models[i].get_conservative_lipschitz_constant(q, r=r)

        return L


    def get_lipschitz_constant(self, c1, c2):
        '''
        Method 3
        '''

        if c1 > c2: c1, c2 = c2, c1
        i = 0
        for c1_ in range(c1 + 1):
            for c2_ in range(c1 + 1, self.num_classes):
                if c1_ == c1 and c2_ == c2: break
                i += 1
        
        L = self.models[i].get_lipschitz_constant()

        return L

    def get_lipschitz_constant_autograd(self, c1, c2):
        '''
        Method 3 with AutoGrad
        '''

        if c1 > c2: c1, c2 = c2, c1
        i = 0
        for c1_ in range(c1 + 1):
            for c2_ in range(c1 + 1, self.num_classes):
                if c1_ == c1 and c2_ == c2: break
                i += 1
        
        L = self.models[i].get_lipschitz_constant_autograd()

        return L

    def get_lipschitz_constant_gd(self, c1, c2):
        '''
        Method 2: Global lipschitz constant

        K = max_x ||\nabla r(x)|| 
        '''

        if c1 > c2: c1, c2 = c2, c1
        i = 0
        for c1_ in range(c1 + 1):
            for c2_ in range(c1 + 1, self.num_classes):
                if c1_ == c1 and c2_ == c2: break
                i += 1
        
        X, loss_history = self.models[i].get_lipschitz_constant_gd()

        return X, loss_history

    def get_conservative_lipschitz_constant_gd(self, c1, c2):
        '''
        Method 3: Conservative lipschitz constant
        '''

        if c1 > c2: c1, c2 = c2, c1
        i = 0
        for c1_ in range(c1 + 1):
            for c2_ in range(c1 + 1, self.num_classes):
                if c1_ == c1 and c2_ == c2: break
                i += 1
        
        X, L, loss_history = self.models[i].get_conservative_lipschitz_constant_gd()

        return X, L, loss_history

    
    def get_eps(self, c1, c2, q, L):

        if c1 > c2: c1, c2 = c2, c1
        i = 0
        for c1_ in range(c1 + 1):
            for c2_ in range(c1 + 1, self.num_classes):
                if c1_ == c1 and c2_ == c2: break
                i += 1

        eps = self.models[i].get_eps(q, L)

        return eps


if __name__ == '__main__':

    import matplotlib.pyplot as plt

    original_dim = 2
    dim = 4000

    encoder = FHRREncoder(original_dim, dim)

    X1 = torch.empty(10, original_dim).normal_(mean=0,std=1)
    X2 = torch.empty(10, original_dim).normal_(mean=10,std=1)

    print(X1)
    print(X2)

    X1_encoded = encoder(X1)
    X2_encoded = encoder(X2)

    print(X1_encoded)
    print(X2_encoded)

    print(X1_encoded.shape)
    print(X2_encoded.shape)

    X = torch.concatenate((X1, X2), dim=0)
    y = torch.tensor([0 for _ in range(len(X1))] + [1 for _ in range(len(X2))])
    print(X.shape, y.shape, y)

    model = ClassificationModel(encoder, in_dim=original_dim, dim=dim)
    model.bundling(X, y)
    print(model.class_hvs1)
    print(model.class_hvs1.shape, model.class_hvs2.shape)

    # print('Method1', model.get_linear_approximation(X[0]))
    # print('Method1 (autograd)', model.get_linear_approximation_autograd(X[0]))
    # print('Method2', model.get_conservative_lipschitz_constant(X[0]))
    # print('Method3', model.get_lipschitz_constant())
    # print('Method3 (autograd)', model.get_lipschitz_constant_autograd())

    # max_X, history = model.get_lipschitz_constant_gd()
    # plt.plot(history)
    # plt.ylabel('$|| \\nabla r(x) ||$')
    # plt.xlabel('Iteration')
    # plt.show()

    # print('Global GD', history[-1], max_X)
    # print('Test', model.get_linear_approximation_autograd(max_X))

    max_X, L, history = model.get_conservative_lipschitz_constant_gd()
    plt.plot(history)
    plt.ylabel('$\\alpha$')
    plt.xlabel('Iteration')
    plt.show()

    print('Conservative GD', L, history[-1], max_X)
    print('Test', model.get_conservative_lipschitz_constant(max_X))
