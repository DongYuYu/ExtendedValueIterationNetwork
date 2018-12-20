import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra, depth_first_order


class gridworld:
    """A class for making gridworlds"""
    def __init__(self, image, targetx, targety):
        self.image = image


        self.n_row = image.shape[0]
        self.n_col = image.shape[1]
        self.obstacles = []
        self.freespace = []
        self.targetx = targetx
        self.targety = targety
        self.G = []
        self.W = []
        self.R = []
        self.P = []
        self.A = []
        self.n_states = 0
        self.n_actions = 0
        self.state_map_col = []
        self.state_map_row = []
        self.set_vals()


    def set_vals(self):
        # Setup function to initialize all necessary
        #  data
        row_obs, col_obs = np.where(self.image == 0)
        row_free, col_free = np.where(self.image != 0)
        self.obstacles = [row_obs, col_obs]
        self.freespace = [row_free, col_free]

        n_states = self.n_row * self.n_col
        n_actions = 8
        self.n_states = n_states
        self.n_actions = n_actions

        p_n = np.zeros((self.n_states, self.n_states))
        p_s = np.zeros((self.n_states, self.n_states))
        p_e = np.zeros((self.n_states, self.n_states))
        p_w = np.zeros((self.n_states, self.n_states))
        p_ne = np.zeros((self.n_states, self.n_states))
        p_nw = np.zeros((self.n_states, self.n_states))
        p_se = np.zeros((self.n_states, self.n_states))
        p_sw = np.zeros((self.n_states, self.n_states))

        R = -1 * np.ones((self.n_states, self.n_actions))
        R[:,4:self.n_actions] = R[:,4:self.n_actions] * np.sqrt(2)
        target = np.ravel_multi_index([self.targetx,self.targety], 
            (self.n_row,self.n_col), order='F')
        R[target,:] = 0
        
        for row in range(0, self.n_row):
            for col in range(0, self.n_col):

                curpos = np.ravel_multi_index([row,col], 
                    (self.n_row,self.n_col), order='F')

                rows, cols = self.neighbors(row, col)

                neighbor_inds = np.ravel_multi_index([rows,cols], 
                    (self.n_row,self.n_col), order='F')

                p_n[curpos, neighbor_inds[0]] = p_n[curpos, 
                                                    neighbor_inds[0]] + 1
                p_s[curpos, neighbor_inds[1]] = p_s[curpos, 
                                                    neighbor_inds[1]] + 1
                p_e[curpos, neighbor_inds[2]] = p_e[curpos, 
                                                    neighbor_inds[2]] + 1
                p_w[curpos, neighbor_inds[3]] = p_w[curpos, 
                                                    neighbor_inds[3]] + 1
                p_ne[curpos, neighbor_inds[4]] = p_ne[curpos, 
                                                    neighbor_inds[4]] + 1
                p_nw[curpos, neighbor_inds[5]] = p_nw[curpos, 
                                                    neighbor_inds[5]] + 1
                p_se[curpos, neighbor_inds[6]] = p_se[curpos, 
                                                    neighbor_inds[6]] + 1
                p_sw[curpos, neighbor_inds[7]] = p_sw[curpos, 
                                                    neighbor_inds[7]] + 1

        G = np.logical_or.reduce((p_n, p_s, p_e, p_w, 
            p_ne, p_nw, p_se, p_sw))

        W = np.maximum(np.maximum(np.maximum(np.maximum(np.maximum(
            np.maximum(np.maximum(p_n, p_s), p_e), p_w), np.sqrt(2) * p_ne), 
            np.sqrt(2) * p_nw), np.sqrt(2) * p_se), np.sqrt(2) * p_sw)
#        print("W")
#        print (W.shape)
        non_obstacles = np.ravel_multi_index([self.freespace[0], 
            self.freespace[1]], (self.n_row,self.n_col), order='F')

        non_obstacles = np.sort(non_obstacles)
        p_n = p_n[non_obstacles,:]
        p_n = np.expand_dims(p_n[:, non_obstacles], axis=2)
        p_s = p_s[non_obstacles,:]
        p_s = np.expand_dims(p_s[:, non_obstacles], axis=2)
        p_e = p_e[non_obstacles,:]
        p_e = np.expand_dims(p_e[:, non_obstacles], axis=2)
        p_w = p_w[non_obstacles,:]
        p_w = np.expand_dims(p_w[:, non_obstacles], axis=2)
        p_ne = p_ne[non_obstacles,:]
        p_ne = np.expand_dims(p_ne[:, non_obstacles], axis=2)
        p_nw = p_nw[non_obstacles,:]
        p_nw = np.expand_dims(p_nw[:, non_obstacles], axis=2)
        p_se = p_se[non_obstacles,:]
        p_se = np.expand_dims(p_se[:, non_obstacles], axis=2)
        p_sw = p_sw[non_obstacles,:]
        p_sw = np.expand_dims(p_sw[:, non_obstacles], axis=2)
        G = G[non_obstacles, :]
        G = G[:, non_obstacles]
        W = W[non_obstacles, :]
        W = W[:, non_obstacles]
        R = R[non_obstacles, :]

        P = np.concatenate((p_n, p_s, p_e, p_w, 
            p_ne, p_nw, p_se, p_sw), axis=2)

        self.G = G
        self.W = W
        self.P = P
        self.R = R
        state_map_col, state_map_row = np.meshgrid(np.arange(0,self.n_col), 
            np.arange(0, self.n_row))
        self.state_map_col = state_map_col.flatten('F')[non_obstacles]
        self.state_map_row = state_map_row.flatten('F')[non_obstacles]


    def get_graph(self):
        # Returns graph
        G = self.G
        W = self.W[self.W != 0]
        return G, W


    def get_graph_inv(self):
        # Returns transpose of graph
        G = self.G.T
        W = self.W.T
        return G, W


    def val_2_image(self, val):
        # Zeros for obstacles, val for free space
        im = np.zeros((self.n_row, self.n_col))
        im[self.freespace[0], self.freespace[1]] = val
        return im


    def get_value_prior(self):
        # Returns value prior for gridworld
        s_map_col, s_map_row = np.meshgrid(np.arange(0,self.n_col), 
            np.arange(0, self.n_row))
        im = np.sqrt(np.square(s_map_col - self.targety) 
            + np.square(s_map_row - self.targetx))
#        print("valuep.................................................................................")
#        print(im)
        return im


    def get_reward_prior(self):
        # Returns reward prior for gridworld
        im = -1 * np.ones((self.n_row, self.n_col))
        im[self.targetx, self.targety] = 10
#        print("reward")
#        print(im)
        return im


    def t_get_reward_prior(self):
        # Returns reward prior as needed for
        #  dataset generation
        im = np.zeros((self.n_row, self.n_col))
        im[self.targetx, self.targety] = 10
        return im


    def get_state_image(self, row, col):
        # Zeros everywhere except [row,col]
        im = np.zeros((self.n_row, self.n_col))
        im[row, col] = 1
        return im


    def map_ind_to_state(self, row, col):
        # Takes [row, col] and maps to a state
        rw = np.where(self.state_map_row == row)
        cl = np.where(self.state_map_col == col)
        if len(np.intersect1d(rw, cl)) != 0 and (rw is not None or cl is not None):
#            print(len(np.intersect1d(rw, cl)))
            return np.intersect1d(rw, cl)[0]
        else:
            return None


    def get_coords(self, states):
        # Given a state or states, returns
        #  [row,col] pairs for the state(s)
        non_obstacles = np.ravel_multi_index(
            [self.freespace[0], self.freespace[1]], 
            (self.n_row,self.n_col), order='F')
#        print("free")
#        print(self.freespace[0])
#        print("free1")
#        print(self.freespace[1])
        non_obstacles = np.sort(non_obstacles)
 #       print("non_obstacles from get_coords")
  #      print(non_obstacles)
        states = states.astype(int)
        r, c = np.unravel_index(non_obstacles[states], 
            (self.n_col, self.n_row), order='F')
   #     print("r")
    #    print(r)
     #   print("c")
      #  print(c)
        return r, c


    def rand_choose(self, in_vec):
        # Samples 
        if len(in_vec.shape) > 1:
            if in_vec.shape[1] == 1:
                in_vec = in_vec.T
        temp = np.hstack((np.zeros((1)), np.cumsum(in_vec))).astype('int')
        q = np.random.rand()
        x = np.where(q > temp[0:-1])
        y = np.where(q < temp[1:])
        return np.intersect1d(x, y)[0]


    def next_state_prob(self, s, a):
        # Gets next state probability for
        #  a given action (a)
        if hasattr(a, "__iter__"):
            p = np.squeeze(self.P[s, :, a])
        else:
            p = np.squeeze(self.P[s, :, a]).T
        return p


    def sample_next_state(self, s, a):
        # Gets the next state given the
        #  current state (s) and an 
        #  action (a)
        vec = self.next_state_prob(s, a)
        result = self.rand_choose(vec)
        return result


    def get_size(self):
        # Returns domain size
        return self.n_row, self.n_col


    def north(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.max([row-1, 0])
        new_col = col
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def northeast(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.max([row - 1, 0])
        new_col = np.min([col + 1, self.n_col - 1])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def northwest(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.max([row - 1, 0])
        new_col = np.max([col - 1, 0])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def south(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.min([row + 1, self.n_row - 1])
        new_col = col
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def southeast(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.min([row + 1, self.n_row - 1])
        new_col = np.min([col + 1, self.n_col - 1])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def southwest(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = np.min([row + 1, self.n_row - 1])
        new_col = np.max([col - 1, 0])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def east(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = row
        new_col = np.min([col + 1, self.n_col - 1])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def west(self, row, col):
        # Returns new [row,col]
        #  if we take the action
        new_row = row
        new_col = np.max([col - 1, 0])
        if self.image[new_row, new_col] == 0:
            new_row = row
            new_col = col
        return new_row, new_col


    def neighbors(self, row, col):
        # Get valid neighbors in all valid directions
        rows, cols = self.north(row, col)
        new_row, new_col = self.south(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.east(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.west(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.northeast(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.northwest(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.southeast(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        new_row, new_col = self.southwest(row, col)
        rows, cols = np.append(rows, new_row), np.append(cols, new_col)
        return rows, cols

    def escape(self, rowS, colS, rowO, colO):

        if abs(rowS-rowO) + abs(colS-colO) == 2:
 #           print("first")
            newR = rowS + rowS - rowO
            newC = colS + colS - colO
            s = self.map_ind_to_state(newR, newC)

            if s is not None:
                #print("111111111111111111111111")
                return s

            s = self.map_ind_to_state(newR, colS)
            
            if s is not None:
#                print("2222222222222222222222222")
                return s

            s = self.map_ind_to_state(rowS, newC) 

            if s is not None:
 #               print("3333333333333333333333333333")
                return s
            else:
                return self.map_ind_to_state(rowS, colS)



        elif abs(rowS-rowO) + abs(colS-colO) == 1:
#            print("second")
            newR = rowS + rowS - rowO
            newC = colS + colS - colO
            s = self.map_ind_to_state(newR, newC)
            if s is not None:
                return s

            if rowS-rowO == 0:
  #              print("000000000000000000000000000000")
                s = self.map_ind_to_state(newR + 1, newC)
                if s is not None:
                    return s
                s = self.map_ind_to_state(newR - 1, newC)
                if s is not None:
                    return s
                
            elif colS-colO == 0:
   #             print("11111111111111111")
                s = self.map_ind_to_state(newR, newC + 1)
                if s is not None:
                    return s
                s = self.map_ind_to_state(newR, newC - 1)
                if s is not None:
                    return s

            
            return self.map_ind_to_state(rowS, colS)
            
        else:
    #        print("someting wrong in escape")
            return self.map_ind_to_state(rowS, colS)


def trace_path(pred, source, target):
    # traces back shortest path from
    #  source to target given pred
    #  (a predicessor list)
#    print("inside")
    max_len = 1000
    path = np.zeros((max_len, 1))
    i = max_len - 1
    path[i] = target
    while path[i] != source and i > 0:
        try:
#            print("i")
#            print(i)
#            print("path[i]")
#            print(path[i])
#            print("pre")
#            print(pred[int(path[i])]) 
            path[i-1] = pred[int(path[i])]
           
            i -= 1
#            print("try")
        except Exception as e:
#            print(e)
            return []

    if i >= 0:
        path = path[i:]
    else:
        path = None
    return path
'''

def sample_trajectory(M, n_states):
    # Samples trajectories from random nodes
    #  in our domain (M)
    G, W = M.get_graph_inv()
    print("G")
    print(G)
    print("W")
    print(W)
    N = G.shape[0]
    print("N")
    print(N)
   # print("shape")
   # print(G.shape)
    if N >= n_states:
        rand_ind = np.random.permutation(N)
    else:
        rand_ind = np.tile(np.random.permutation(N), (1,10))
#    print("random")
#    print(rand_ind)
    init_states = rand_ind[0:n_states].flatten()

    r, c = M.get_coords(init_states[0])
    r1, c1 = M.neighbors(r,c)
    l = len(r1)
    init_statesO = M.map_ind_to_state(r1[l-1], c1[l-1])
    
    print("init")
    print(init_states)


#    init_statesO = rand_ind[2:4].flatten()
    print("init1")
    print(init_statesO)
    goal_s = M.map_ind_to_state(M.targetx, M.targety)
    states = []
    states_xy = []
    states_one_hot = []
    # Get optimal path from graph
    g_dense = W
    print("sparse")
#    print(g_dense)
    g_masked = np.ma.masked_values(g_dense,0)
    g_sparse = csr_matrix(g_dense)
    print(g_sparse)
    d, pred = dijkstra(g_sparse, indices=goal_s, return_predecessors=True)
    print("gs")
    print(goal_s)
#    print("-----------------------d")
#    print(d)
#    print("---------------------pred")
#    print(pred)
    for i in range(n_states):
        print("lookhere")
        path = trace_path(pred, goal_s, init_states[i])
        print("finish")
        path = np.flip(path, 0)
        states.append(path)
        print("Orighinal path")
        print(path)
#        print("for")
#        print(i)
#        print(path)

    count = 0
    states1=[]
    states1_xy=[]
    for state in states:

        L = len(state)
        for i in range(L):
            s = np.squeeze(state[i])
            d1, pred1 = dijkstra(g_sparse, indices=s, return_predecessors=True)

            
#            print("look")
            path1 = trace_path(pred1, s, init_statesO)
#            print("finish")
            path1 = np.flip(path1, 0)
#            print("length")
#            print(len(path1))
            states1.append(path1)
            #print("pred1")
            #print(pred1)
#            print("path1")
#            print(path1)
#            print("statei")
#            print(state[i])
#            print("initiateO")
#            print(init_statesO)
            if len(path1) > 1:
                init_statesO = path1[1]
        print("state")
        print(type(state))
        print(len(state))
        print(state)
        r, c = M.get_coords(state)
        row_m = np.zeros((L, M.n_row))
        col_m = np.zeros((L, M.n_col))
        for i in range(L):
            row_m[i, r[i]] = 1
            col_m[i, c[i]] = 1
        states_one_hot.append(np.hstack((row_m, col_m)))
        states_xy.append(np.hstack((r, c)))
        print("states_xy123")
        print(states_xy)
#        print("states_one_hot")
#        print(states_one_hot)
    count1 = 0
    for state1 in states1:
        r1, c1 = M.get_coords(state1)
        states1_xy.append(np.hstack((r1,c1)))
        print("opathfromini1 to state:")
        print(count1)
        count1 = count1 + 1
        print(np.hstack((r1, c1)))
    print("state1")
    print(states1_xy)
    print("-------------------------------------------------")
    return states_xy, states_one_hot

'''
def sample_trajectory(M, n_states):
    # Samples trajectories from random nodes
    #  in our domain (M)
    G, W = M.get_graph_inv()
#    print("G")
#    print(G)
#    print("W")
#    print(W)
    N = G.shape[0]
    print("N")
    print(N)
   # print("shape")
   # print(G.shape)
    if N >= n_states:
        rand_ind = np.random.permutation(N)
        rand_ind1 = np.random.permutation(N)
   #     print("random")
   #     print(rand_ind)
   #     print(rand_ind1)
    else:
        rand_ind = np.tile(np.random.permutation(N), (1,10))
        rand_ind1 = np.tile(np.random.permutation(N), (1,10))
#    print("random")
#    print(rand_ind)
    init_states = rand_ind[0:n_states].flatten()

 

#    init_statesO = rand_ind[2:4].flatten()
#    print("init1")
#    print(init_statesO)
    goal_s = M.map_ind_to_state(M.targetx, M.targety)
    states = []
    states_xy = []
    states_one_hot = []
    # Get optimal path from graph
    g_dense = W
#    print("sparse")
#    print(g_dense)
    g_masked = np.ma.masked_values(g_dense,0)
    g_sparse = csr_matrix(g_dense)
#    print(g_sparse)
    d, pred = dijkstra(g_sparse, indices=goal_s, return_predecessors=True)
#    print("gs")
#    print(goal_s)

    init_states1 = rand_ind1[0:N].flatten()
#    print(init_states1)
    init_states0=[]
    for i in range(n_states):
        
        Tcsr = depth_first_order(g_sparse, init_states[i], directed=False, return_predecessors=False)

        dis = np.random.permutation(len(Tcsr))
        r, c = M.get_coords(goal_s)
        r0, c0 = M.get_coords(init_states[i])

#        print(Tcsr)
        for j in range(len(dis)):
            ro, co = M.get_coords(Tcsr[dis[j]])
            dr = abs(r-r0)
            dc = abs(c-c0)
            dro = abs (ro-r)
            drc = abs (co -c)
            if Tcsr[dis[j]] != goal_s and dr>dro and dc>drc:
                print("dr")
                print(dr)
                print(dc)
                print("dio")
                print(dro)
                print(drc)
                init_states0.append(Tcsr[dis[j]])
                break;
       # Tcsr1 = Tcsr.todense()
      #  print("dfs")
      #  print(Tcsr1.shape)
      
 #       for j in range(len(init_states1)-1, -1, -1):
 #           print("init[i]")
 #           print(init_states[i])
 #           print("init[j]")
 #           print(init_states1[j])
         #   print(Tcsr1)
          #  print(Tcsr1[0])
           # print(Tcsr1[0][0])
            
#            if init_states1[j] != goal_s and (Tcsr[init_states[i],init_states1[j]] != 0 or Tcsr[init_states1[j], init_states[i]] != 0):
  #              print("in here")
#                init_states0.append(init_states1[j])
#                break
            
        
            
        '''
    r, c = M.get_coords(init_states[0])
    r1, c1 = M.neighbors(r,c)
    l = len(r1)
    init_statesO = M.map_ind_to_state(r1[l-1], c1[l-1])
    
    print("init")
    print(init_states)
        
    r, c = M.get_coords(init_states[0])
    r1, c1 = M.neighbors(r,c)
    l = len(r1)
    init_statesO = M.map_ind_to_state(r1[l-1], c1[l-1])
    
    print("init")
    print(init_states)
'''
#    print("-----------------------d")
#    print(d)
#    print("---------------------pred")
#    print(pred)
    valuef = []
    for i in range(n_states):
       # print("lookhere")
        path = trace_path(pred, goal_s, init_states[i])
       # print("finish")
        path = np.flip(path, 0)
        states.append(path)
   #     print("Orighinal path")
   #     print(path)
#        print("for")
#        print(i)
#        print(path)

    count = 0
    states1=[]
    pathO=[]
    states1_xy=[]
    count1 = 0
    
    v=[]
    for state in states:
        v= []
    #    print("init_states0")
    #    print(len(init_states0))
        if count1 in range(len(init_states0)):
            init_statesO = init_states0[count1]
            count1 = count1 + 1
        elif len(init_states0) == 0 and len(state)>=1:
            init_statesO = state[0]
        else:
            init_statesO = init_states[0]  #problem
        L = len(state)
        for i in range(L):
            s = np.squeeze(state[i])
  #          print("=====================================")
  #          print("indice")
  #          print(s)
            d1, pred1 = dijkstra(g_sparse, indices=s, return_predecessors=True)

            pathO.append(init_statesO)
#            print("look")
            path1 = trace_path(pred1, s, init_statesO)
#            print("finish")
            path1 = np.flip(path1, 0)
#            print("length")
#            print(len(path1))
            #states1.append(path1)
            #print("pred1")
            #print(pred1)
#            print("path1")
#            print(path1)
#            print("statei")
#            print(state[i])
#            print("initiateO")
#            print(init_statesO)
            if len(path1) == 0:
                path1=[]
                path1.append(init_statesO)
                
            value = M.get_reward_prior()
            r0, c0 = M.get_coords(path1[0])
            r, c = M.neighbors(r0, c0)
            #print("rrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrrr")
            #print(r)
            value[r, c] = -3
            value[r0,c0] = -10
     #       print("value??????????????????????????????????????????????????????")
     #       print(value)
            v.append(value)
            if len(path1) > 2:
                init_statesO = path1[1]

            elif i < L-1 and len(path1) > 1:
                 init_statesO = path1[1]
                 r0, c0 = M.get_coords(path1[0])
                 r, c = M.get_coords(path1[1])
                 state[i + 1] = M.escape(r,c,r0,c0)
 #                print("escape")
              #   r1 = r + r - r0
              #   c1 = c + c - c0
              #   s = M.map_ind_to_state(r1, c1)
              #   if s is not None:
      #              print("changes=====================================================to")
#                    print(r1, c1)
               #     state[i+1] = s
               #  else:
       #             print("remain==========================================================")
                #    state[i+1] = state[i]
            elif i < L-1 and len(path1) == 1:
                 state[i + 1] = state[i]
                 
#                print("===============================")
#                print("state[i+1:]")
#                print(state[i+1:])
#                print("state[i]")
#                print(state[i])
#                state = state[:i]
#                pathO[i:] = [init_statesO] * (1)
#                v[i:] = [value] * (1)

   #             state[i+1:] = state[i]
   #             pathO[i+1:] = [init_statesO] * (L - i)
   #             v[i+1:] = [value] * (L - i)
   #             break

        pathO1 = np.asarray(pathO).reshape(-1,1)
       # print("stateO")
       # print(pathO1)
        states1.append(pathO1)
        pathO=[]
       # print("state")
       # print(type(state))
       # print(len(state))
       # print(state)
        r, c = M.get_coords(state)
        row_m = np.zeros((L, M.n_row))
        col_m = np.zeros((L, M.n_col))
        L = len(state)
        for i in range(L):
            row_m[i, r[i]] = 1
            col_m[i, c[i]] = 1
        states_one_hot.append(np.hstack((row_m, col_m)))
        states_xy.append(np.hstack((r, c)))
        #print("states_xy123")
        #print(states_xy)
        valuef.append(v)
#        print("states_one_hot")
#        print(states_one_hot)
#    count1 = 0
   # print("valuef")
   # print(valuef)
    if len(valuef) == 0 :
        value = M.get_reward_prior()
    #    print("valuef=0")
        
        v.append(value)
        valuef.append(v)
     #   print(valuef)
    for state1 in states1:
       # print("state1")
       # print(state1)
        r1, c1 = M.get_coords(state1)
        states1_xy.append(np.hstack((r1,c1)))
       # print("opath:")
        #print(count1)
        #count1 = count1 + 1
       # print(np.hstack((r1, c1)))
    #print("state1")
    #print(states1_xy)

    
    
    
#    print("-------------------------------------------------")
#    print("-----------------------------------------------------")
#    print("state")
#    print(states_xy[0])
#    print("O")
#    print(states1_xy[0])
    
    #print("lenofvanluf")
    #print(len(valuef[0]))
    #print("lengofstate")
    #print(len(states_xy[0]))
    assert len(valuef) == len(states1_xy)
    assert len(states1_xy) == len(states_xy)
    assert len(states_one_hot) == len(states1_xy)
    return valuef, states1_xy, states_xy, states_one_hot
