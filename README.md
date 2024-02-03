## Simulation-free flow matching with constraints.
Based on <a href="https://arxiv.org/abs/2311.13443">Guided Flows for Generative Modeling and Decision Making</a>, Q. Zheng, M. Le, N. Shaul, Y. Lipman, A. Grover, R. T. Q. Chen 2023.

We can enforce a constraint on our learned vector field $f(x,t)$ via barrier functions to acheive forward set-invariance wrt the set $\mathcal{C} = \\{x \in \mathbb{R}^d : h(x) \geq 0 \\}$. This can be done by approximately enforcing the following condition on the learned vector-field:

$$
\begin{align*}
\nabla_x h(x)^\top f(x,t) \geq -\alpha h(x) \quad \forall (x,t) \in \mathcal{C}\times [0,1]
\end{align*}
$$


Here we are matching the time-varying vector-field: $\dot x(t) = x_1-x_0$ with no guidance. In the figure below we manage to add a constraint on the flow by using a barrier function on the vector field itself. Still simultion-free!

### Example 1: Box constraint on the x-coordinate $|x| < 0.5$

![alt text](https://github.com/AaronHavens/FlowMatching/blob/main/assets/linear_circle_flow_wall.gif?raw=true)



### Example 2: Circular hole constraint $x^2 + (y-0.5)^2 \geq 0.25^2$

![alt text](https://github.com/AaronHavens/FlowMatching/blob/main/assets/linear_circle_flow_hole.gif?raw=true)


Notice that because we are matching a linear interpolant $x_t = (1-t)x_0 - t x_1$, the flow lines will not go behind the hole constraint. We can change that.

### Example 3: Bezier Interpolants $x^2 + (y-0.5)^2 \geq 0.25^2$

Rather than using a linear interpolant, we can use nonlinear Bezier paths. In particular, we use a cubic Bezier curve and learn the control points from data. Given end points $x_0, x_1$ we parameterize a simple NN $f_\theta$ to give us the intermediate control points $(z_0,z_1) = f_\theta(x_0, x_1)$. Then our interpolants and cooresponding velocity fields are given by:

$$
\begin{align*}
x_t = (1-t)^3 x_0 + 3 (1-t)^2 t z_0 + 3 (1-t)  t^2 z_1 + t^3 * x_1
\dot x_t = 3 (1-t)^2 (z_0-x_0) + 6 (1-t) t (z_1-z_0) + 3 t^2 (x_1-z_1)
\end{align*}
$$

![alt text](https://github.com/AaronHavens/FlowMatching/blob/main/assets/bezier_circle_flow_hole.gif?raw=true)

With nonlinear interpolants, the flow lines can now go around the hole constraint :).