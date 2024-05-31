class FQA:
    
    def __init__(self, A, b, dt):
        # construct Hamiltonians
        self.Hp = self.get_problem(A, b)
        self.Hm = self.get_mixer()
        self.Hd = self.get_driver()
        
        # construct variables of interest
        self.dt = dt
        self.sol = eigh(self.Hp)[1][:,0]
        self.state = expm(-1j*5*self.Hm) @ eigh(self.Hd)[1][:,0]  # 
start in ground state of driver, evolved by mixer
        # self.state = expm(-1j*10*self.Hp) @ eigh(self.Hd)[1][:,0]  
# start in ground state of driver, evolved by problem
        self.params = deque()
        self.overlaps = deque()
        self.objvals = deque()
    
    
    def evolve(self, m):
        '''run the fqa for m layers'''
        for _ in tqdm(range(m)):
            self.one_layer()
    
    def rand_evolve(self, m):
        '''run the randomized fqa for m layers'''
        for _ in tqdm(range(m)):
            self.one_rand_layer()
    
    
    def one_layer(self):
        '''evolve one layer of fqa'''
        self.params.append(self.get_param(self.Hd))
        # self.params.append(self.get_param(self.Hd) + 
2*(np.random.rand() < 0.7)) # random kicks
        self.state = expm(-1j*self.dt*self.params[-1]*self.Hd) @ 
self.state
        self.params.append(self.get_param(self.Hm))
        # self.params.append(self.get_param(self.Hm) + 
2*(np.random.rand() < 0.7)) # random kicks
        self.state = expm(-1j*self.dt*self.params[-1]*self.Hm) @ 
self.state
        self.state = expm(-1j*self.dt*self.Hp) @ self.state
        self.objvals.append(self.get_objval())
        self.overlaps.append(np.abs(np.dot(np.conjugate(self.sol), 
self.state))**2)
    
    def one_rand_layer(self):
        '''evolve one layer of the randomized fqa'''
        Vk = random_clifford(self.n).to_matrix()
        curr_H = Vk @ self.Hd @ np.conjugate(Vk).T
        self.params.append(self.get_param(curr_H))
        self.state = expm(-1j*self.dt*self.params[-1]*curr_H) @ 
self.state
        self.objvals.append(self.get_objval())
        self.overlaps.append(np.abs(np.dot(np.conjugate(self.sol), 
self.state))**2)
    
    
    def get_param(self, H):
        '''get parameter for Hamiltonian H with current state'''
        return -1j * np.dot(np.conjugate(self.state), (H @ self.Hp - 
self.Hp @ H) @ self.state)
    
    
    def get_objval(self):
        '''get value of objective function at current state'''
        return np.dot(np.conjugate(self.state), self.Hp @ 
self.state)
    
    
    def get_mixer(self):
        '''construct mixer Hamiltonian'''
        n = self.n
        X = sp.csc_matrix([[0,1],[1,0]])
        Hm = sp.kron(X, sp.csc_matrix(sp.eye(2**(n-1))))
        for j in range(1, n):
            Hm += reduce(sp.kron, [sp.csc_matrix(sp.eye(2**j)), X, 
sp.csc_matrix(sp.eye(2**(n-j-1)))])
        return np.array(Hm.todense())
    
    
    def get_driver(self):
        '''construct driver Hamiltonian'''
        n = self.n
        # initialize
        Z = sp.csc_matrix([[1,0],[0,-1]])
        Y = sp.csc_matrix([[0, -1j],[1j, 0]])
        Hd = reduce(sp.kron, [Z, Z, 
sp.csc_matrix(sp.eye(2**(n-2)))])

        # loop over qubits
        for j in range(1, n-1):
            Hd += reduce(sp.kron, 
[sp.csc_matrix(sp.eye(2**j)),Z,Z,sp.csc_matrix(sp.eye(2**(n-j-2)))])

        # add final connections
        Hd += reduce(sp.kron, [Z,sp.csc_matrix(sp.eye(2**(n-2))),Z])
        Hd += reduce(sp.kron, [Y, Y, 
sp.csc_matrix(sp.eye(2**(n-2)))])
        return np.array(Hd.todense())
    
    
    def get_problem(self, A, b):
        '''construct problem Hamiltonian'''
        self.n = int(np.log2(len(b)))
        return np.conjugate(A.T) @ (np.eye(2**self.n) - 
np.outer(b,b)) @ A
