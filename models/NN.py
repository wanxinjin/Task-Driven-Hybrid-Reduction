from casadi import *


# #############################################
# this class implements  neural dynamics models
###############################################
class NNDyn:

    def __init__(self, n_x, n_u, n_z=None, n_hidden=None):

        # dims
        self.n_x = n_x
        self.n_u = n_u

        if n_z is None:
            self.n_z = n_x
        else:
            self.n_z = n_z

        # variables
        self.x = SX.sym('x', self.n_x)
        self.u = SX.sym('u', self.n_u)
        self.z = SX.sym('z', self.n_z)

        # NN mapping from (x,u) to (z)
        nn_in = vertcat(self.x, self.u)

        # the layers
        if n_hidden is None:
            n_hidden = []

        n_hidden = [nn_in.numel()] + n_hidden + [self.n_z]

        nn_param = []
        actv = nn_in
        for i in range(len(n_hidden) - 2):
            # define the weights and bias for the layer i
            Wi = SX.sym('W' + str(i), n_hidden[i + 1], n_hidden[i])
            bi = SX.sym('b' + str(i), n_hidden[i + 1])
            nn_param.append(vec(Wi))
            nn_param.append(bi)

            # do the mapping to the next layer using the activation function
            actv = Wi @ actv + bi
            # activation function (swish)
            # actv = fmax(actv, 0.0)
            actv = actv / (1 + exp(actv))
            # actv = 2 / (1 + exp(-2*actv))-1

        # the final layer
        Wo = SX.sym('Wo', n_hidden[-1], n_hidden[-2])
        bo = SX.sym('bo', n_hidden[-1])
        nn_param.append(vec(Wo))
        nn_param.append(bo)
        # do the mapping to the next layer using the activation function
        actv = Wo @ actv + bo

        # construct nn
        self.nn_param = vcat(nn_param)
        self.n_nnparam = self.nn_param.numel()
        self.nn_forward_fn = Function('nn_forward_fn', [self.nn_param, self.x, self.u], [actv])

    # full dynamics x_next = expr_fn(x,z), where z=nn(nn_param, x,u)
    def initDyn(self, expr_fn=None):

        if not hasattr(self, 'nn_forward_fn'):
            assert False, "please first use initNN to initialize the dynamics"

        if expr_fn is None:
            self.expr_fn = Function('expr_fn', [self.x, self.z], [self.z])
        else:
            self.expr_fn = expr_fn

        # compose the dynamics
        dyn = self.expr_fn(self.x, self.nn_forward_fn(self.nn_param, self.x, self.u))
        self.aux = self.nn_param
        self.n_aux = self.aux.numel()
        self.dyn_fn = Function('dyn_fn', [self.aux, self.x, self.u], [dyn])

        # check the dims of the dynamics
        assert self.dyn_fn.numel_out(
            0) == self.n_x, 'please check your dynamics, the dims of input and output are different'

        # differentiate the dynamics
        self.jac_dyn2x_fn = Function('jac_dyn2x_fn', [self.aux, self.x, self.u],
                                     [jacobian(self.dyn_fn(self.aux, self.x, self.u), self.x)])
        self.jac_dyn2u_fn = Function('jac_dyn2u_fn', [self.aux, self.x, self.u],
                                     [jacobian(self.dyn_fn(self.aux, self.x, self.u), self.u)])
        self.jac_dyn2aux_fn = Function('jac_dyn2aux_fn', [self.aux, self.x, self.u],
                                       [jacobian(self.dyn_fn(self.aux, self.x, self.u), self.aux)])

    # integrate one-step dynamics and differentiate next state with respect to x,u, aux
    def forwardDiff(self, aux_val, x_val, u_val):

        if not hasattr(self, 'initDyn'):
            assert False, 'please using the initDyn method to initialize your dynamics'

        # compute the next state
        y_val = self.dyn_fn(aux_val, x_val, u_val).full().flatten()

        # compute the diff of dynamics
        jac_y2x_val = self.jac_dyn2x_fn(aux_val, x_val, u_val).full()
        jac_y2u_val = self.jac_dyn2u_fn(aux_val, x_val, u_val).full()
        jac_y2aux_val = self.jac_dyn2aux_fn(aux_val, x_val, u_val).full()

        return dict(y_val=y_val,
                    jac_y2x_val=jac_y2x_val,
                    jac_y2u_val=jac_y2u_val,
                    jac_y2aux_val=jac_y2aux_val)

    # integrate multiple step dynamics and differentiate along the trajectory
    def forwardTrajDiff(self, aux_val, x0, u_traj):

        # rollout the dynamics and compute the state trajectory
        T = u_traj.shape[0]
        x_traj = [x0]

        # this is the `accumulated' gradient of dynamics at each step, w.r.t, aux,
        grad_x2aux_traj = [np.zeros((self.n_x, self.n_aux))]  # since the system is given initial condition

        # we will also differentiate each step
        # (which will be used in recovery matrix below)
        jac_y2x_traj = []
        jac_y2u_traj = []
        jac_y2aux_traj = []

        # rollout and differentiate
        for t in range(T):
            # get the current xt and ut
            ut = u_traj[t]
            xt = x_traj[-1]
            grad_xt2aux = grad_x2aux_traj[-1]

            # solve the next x and autodiff at the same time
            yt = self.dyn_fn(aux_val, xt, ut).full().flatten()
            x_traj.append(yt)

            # differentiate at each single step
            jac_yt2xt = self.jac_dyn2x_fn(aux_val, xt, ut).full()
            jac_yt2ut = self.jac_dyn2u_fn(aux_val, xt, ut).full()
            jac_yt2aux = self.jac_dyn2aux_fn(aux_val, xt, ut).full()

            # store
            jac_y2x_traj.append(jac_yt2xt)
            jac_y2u_traj.append(jac_yt2ut)
            jac_y2aux_traj.append(jac_yt2aux)

            # compute the accumulated gradient of x w.r.t. dyn_aux
            grad_yt2dau_t = jac_yt2xt @ grad_xt2aux + jac_yt2aux
            grad_x2aux_traj.append(grad_yt2dau_t)

        # ready for the return
        x_traj = np.array(x_traj)
        jac_y2x_traj_vstack = np.vstack(jac_y2x_traj)  # note without the  info about x0
        jac_y2u_traj_vstack = np.vstack(jac_y2u_traj)  # note without the  info about x0
        jac_y2aux_traj_vstack = np.vstack(jac_y2aux_traj)  # note without the  info about x0
        grad_x2aux_traj = np.array(grad_x2aux_traj)

        return dict(x_traj=x_traj,
                    u_traj=u_traj,
                    grad_x2aux_traj=grad_x2aux_traj,

                    jac_y2x_traj_vstack=jac_y2x_traj_vstack,
                    jac_y2u_traj_vstack=jac_y2u_traj_vstack,
                    jac_y2aux_traj_vstack=jac_y2aux_traj_vstack,
                    )


# #############################################
# this class implements nn model learning
# #############################################
class NNDynTrainer:

    def __init__(self, nn_dyn: NNDyn, opt_gd, init_aux_val=None):

        self.nn_dyn = nn_dyn
        self.dyn_fn = self.nn_dyn.dyn_fn

        # define variable
        self.n_aux = self.dyn_fn.numel_in(0)
        self.aux = SX.sym('aux', self.n_aux)
        self.n_x = self.dyn_fn.numel_in(1)
        self.x = SX.sym('x', self.n_x)
        self.n_u = self.dyn_fn.numel_in(2)
        self.u = SX.sym('u', self.n_u)

        # output/next state
        self.n_y = self.dyn_fn.numel_out(0)
        self.y = SX.sym('y', self.n_y)

        # define loss (l2 norm) and grad
        # y_w = np.array([0.002, 0.002, 0.002, 0.01, 0.07, 0.13, 0.07, 0.13, 0.09])
        y_w = np.ones(self.n_y)
        loss = 0.5 * dot(self.dyn_fn(self.aux, self.x, self.u) - self.y,
                         diag(1 / (y_w * y_w)) @ (self.dyn_fn(self.aux, self.x, self.u) - self.y))
        grad = gradient(loss, self.aux)
        self.loss_grad_fn = Function('loss_grad_fn', [self.aux, self.x, self.u, self.y], [loss, grad])
        self.loss_fn = Function('loss_fn', [self.aux, self.x, self.u, self.y], [loss])

        # init optimizer
        self.opt_gd = opt_gd

        # init aux
        if init_aux_val is None:
            self.aux_val = np.random.uniform(-0.1, 0.1, self.n_aux)
        else:
            self.aux_val = init_aux_val

    # one-step gradient descent given (x,u,y) batch
    def step(self, x_minibatch, u_minibatch, y_minibatch, disable_update=False):
        # compute the loss and grad
        loss_batch, grad_batch = self.loss_grad_fn(self.aux_val.T, x_minibatch.T, u_minibatch.T, y_minibatch.T)

        loss = loss_batch.full().ravel().mean()
        grad = grad_batch.full().mean(axis=1)

        if not disable_update:
            self.aux_val = self.opt_gd.step(self.aux_val, grad)

        return loss, dict(grad=grad,
                          aux_val=self.aux_val)

    # whole training process
    def train(self,
              x_batch, u_batch, y_batch, eval_ratio=0.2,
              minibatch_size=100, n_epoch=100,
              print_freq=-1):

        n_data = x_batch.shape[0]
        n_traindata = int(n_data * (1 - eval_ratio))
        n_evaldata = n_data - n_traindata

        if minibatch_size > n_traindata:
            minibatch_size = n_traindata

        train_loss_trace = []
        eval_loss_trace = []
        for i in range(n_epoch):

            # shuffling
            all_ids = np.random.permutation(n_data)
            loss_k = []

            # training
            for k in range(int(np.floor(n_traindata / minibatch_size))):
                # walk through the shuffled new data
                minibatch_ids = all_ids[k * minibatch_size: (k + 1) * minibatch_size]
                loss_minibatch, _, = self.step(x_batch[minibatch_ids], u_batch[minibatch_ids], y_batch[minibatch_ids])
                loss_k.append(loss_minibatch)

            # eval
            eval_ids = all_ids[-n_evaldata:]
            eval_loss, _, = self.step(x_batch[eval_ids], u_batch[eval_ids], y_batch[eval_ids], disable_update=True)

            # print
            loss = np.mean(loss_k)
            if (print_freq > 0 and i % print_freq == 0) or i == n_epoch - 1:
                print(f'epoch: {i}, train loss: {loss}, eval loss: {eval_loss}')

            # store
            train_loss_trace.append(loss)
            eval_loss_trace.append(eval_loss)

        return dict(train_loss_trace=np.array(train_loss_trace),
                    eval_loss_trace=np.array(eval_loss_trace),
                    aux_val=self.aux_val)

    # one-step gradient descent given traj batch
    def stepTraj(self, rollout_minibatch, disable_update=False):

        minibatch_size = len(rollout_minibatch)

        grad_batch = []
        loss_batch = []
        for i in range(minibatch_size):
            x_traj = rollout_minibatch[i]['state_traj']
            u_traj = rollout_minibatch[i]['control_traj']

            # weights
            x_w = np.diag(1. / (x_traj * x_traj).mean(axis=0))
            # x_w = np.diag(np.ones(self.n_x))

            # rollout of model
            model_res = self.nn_dyn.forwardTrajDiff(aux_val=self.aux_val, x0=x_traj[0], u_traj=u_traj)
            model_x_traj = model_res['x_traj']
            model_grad_x2aux_traj = model_res['grad_x2aux_traj']

            # gradient and loss
            mse_grad = 0.0
            mse_loss = 0.0
            for t in range(len(model_x_traj)):
                mse_grad += x_w @ (model_x_traj[t] - x_traj[t]) @ model_grad_x2aux_traj[t]
                mse_loss += dot(model_x_traj[t] - x_traj[t], x_w @ (model_x_traj[t] - x_traj[t])).full().item()

            grad_batch.append(mse_grad)
            loss_batch.append(mse_loss)

        grad = np.array(grad_batch).mean(axis=0)
        loss = np.array(loss_batch).mean()

        if not disable_update:
            self.aux_val = self.opt_gd.step(self.aux_val, grad)

        return loss, dict(grad=grad,
                          aux_val=self.aux_val)

    # whole training process
    def trainTraj(self,
                  rollout_batch, eval_ratio=0.2,
                  minibatch_size=10, n_epoch=100,
                  print_freq=-1):

        batch_size = len(rollout_batch)
        batch_size_train = max(int(batch_size * (1 - eval_ratio)), 1)
        batch_size_eval = max(batch_size - batch_size_train, 1)

        if minibatch_size > batch_size_train:
            minibatch_size = batch_size_train

        train_loss_trace = []
        eval_loss_trace = []
        for i in range(n_epoch):

            # shuffling
            all_ids = np.random.permutation(batch_size)
            loss_k = []

            # training
            for k in range(int(np.floor(batch_size_train / minibatch_size))):
                # walk through the shuffled new data
                minibatch_ids = all_ids[k * minibatch_size: (k + 1) * minibatch_size]
                loss_minibatch, _, = self.stepTraj(rollout_minibatch=[rollout_batch[j] for j in minibatch_ids])
                loss_k.append(loss_minibatch)

            # eval
            eval_ids = all_ids[-batch_size_eval:]
            eval_loss, _, = self.stepTraj(rollout_minibatch=[rollout_batch[j] for j in eval_ids],
                                          disable_update=True)

            # print
            loss = np.mean(loss_k)
            if (print_freq > 0 and i % print_freq == 0) or i == n_epoch - 1:
                print(f'epoch: {i}, train loss: {loss}, eval loss: {eval_loss}')

            # store
            train_loss_trace.append(loss)
            eval_loss_trace.append(eval_loss)

        return dict(train_loss_trace=np.array(train_loss_trace),
                    eval_loss_trace=np.array(eval_loss_trace),
                    aux_val=self.aux_val)
