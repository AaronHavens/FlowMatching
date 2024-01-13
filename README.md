## Simulation-free flow matching with constraints.
Based on <a href="https://arxiv.org/abs/2311.13443">Guided Flows for Generative Modeling and Decision Making</a>, Q. Zheng, M. Le, N. Shaul, Y. Lipman, A. Grover, R. T. Q. Chen 2023. 


Here we are matching the linear vector-field: x1-x0 with no guidance. In the figure below we manage to add a constraint on the flow by using a barrier function on the vector field itself. Still simultion-free!

### Example 1: Box constraint on the x-coordinate |x| < 0.5

![alt text](https://github.com/AaronHavens/FlowMatching/blob/main/assets/circle_flow_wall_const.gif?raw=true)



### Example 2: Circular hole constraint x^2 + (y-0.5)^2 >= 0.25^2

![alt text](https://github.com/AaronHavens/FlowMatching/blob/main/assets/circle_flow_hole_const.gif?raw=true)