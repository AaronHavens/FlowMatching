## Simulation-free flow matching with constraints.
Based on <a href="https://arxiv.org/abs/2311.13443">Guided Flows for Generative Modeling and Decision Making</a>, Q. Zheng, M. Le, N. Shaul, Y. Lipman, A. Grover, R. T. Q. Chen 2023.

We can enforce a constraint on our learned vector field $f(x,t)$ via barrier functions to acheive forward set-invariance wrt the set $\mathcal{C} = \\{x \in \mathbb{R}^d : h(x) \geq 0 \\}$. This can be done by approximately enforcing the condition on the learned vector-field:

$$
\begin{align*}
\nabla_x h(x)^\top f(x,t) \geq -\alpha h(x) \quad \forall (x,t) \in \mathcal{C}\times [0,1]
$$


Here we are matching the time-varying vector-field: $\dot x(t) = x_1-x_0$ with no guidance. In the figure below we manage to add a constraint on the flow by using a barrier function on the vector field itself. Still simultion-free!

### Example 1: Box constraint on the x-coordinate $|x| < 0.5$

![alt text](https://github.com/AaronHavens/FlowMatching/blob/main/assets/circle_flow_wall_const.gif?raw=true)



### Example 2: Circular hole constraint $x^2 + (y-0.5)^2 \geq 0.25^2$

![alt text](https://github.com/AaronHavens/FlowMatching/blob/main/assets/circle_flow_hole_const.gif?raw=true)