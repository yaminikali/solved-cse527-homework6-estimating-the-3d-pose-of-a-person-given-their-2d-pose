Download Link: https://assignmentchef.com/product/solved-cse527-homework6-estimating-the-3d-pose-of-a-person-given-their-2d-pose
<br>
In this homework we are going to work on estimating the 3D pose of a person given their 2D pose. Turns out, <a href="https://arxiv.org/pdf/1705.03098.pdf">just regressing the 3D pose coordinates using the 2D pose works pretty well [1] (you can find the paper </a><u><a href="https://arxiv.org/pdf/1705.03098.pdf">here (https://arxiv.or</a></u><a href="https://arxiv.org/pdf/1705.03098.pdf">g</a><u><a href="https://arxiv.org/pdf/1705.03098.pdf">/pdf/1705.03098.pdf)</a></u><a href="https://arxiv.org/pdf/1705.03098.pdf">). In Part One, we are going to work on reproducing the results of the pa</a>per, in Part Two, we are going to try to find a way to handle noisy measurement.

Some Tutorials (PyTorch)

You will be using PyTorch for deep learning toolbox (follow the <u><a href="http://pytorch.org/">link </a></u><a href="http://pytorch.org/">(</a><u><a href="http://pytorch.org/">http://p</a></u><a href="http://pytorch.org/">y</a><u><a href="http://pytorch.org/">torch.or</a></u><a href="http://pytorch.org/">g)</a> for installation).

<a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">For PyTorch beginners, please read this </a><u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">tutorial</a></u>

<u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">(http://p</a></u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">y</a><u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">torch.or</a></u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">g</a><u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">/tutorials/be</a></u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">g</a><u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">inner/deep_learnin</a></u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">g</a><u><a href="http://pytorch.org/tutorials/beginner/deep_learning_60min_blitz.html">_60min_blitz.html)</a></u> before doing your homework.

Feel free to study more tutorials at <u><a href="http://pytorch.org/tutorials/">http://p</a></u><a href="http://pytorch.org/tutorials/">y</a><u><a href="http://pytorch.org/tutorials/">torch.or</a></u><a href="http://pytorch.org/tutorials/">g</a><u><a href="http://pytorch.org/tutorials/">/tutorials/ </a></u><a href="http://pytorch.org/tutorials/">(</a><u><a href="http://pytorch.org/tutorials/">http://p</a></u><a href="http://pytorch.org/tutorials/">y</a><u><a href="http://pytorch.org/tutorials/">torch.or</a></u><a href="http://pytorch.org/tutorials/">g</a><u><a href="http://pytorch.org/tutorials/">/tutorials/)</a></u><a href="http://pytorch.org/tutorials/">.</a>

Find cool visualization here at <u><a href="http://playground.tensorflow.org/">http://play</a></u><a href="http://playground.tensorflow.org/">g</a><u><a href="http://playground.tensorflow.org/">round.tensorflow.or</a></u><a href="http://playground.tensorflow.org/">g</a><u><a href="http://playground.tensorflow.org/"> (http://play</a></u><a href="http://playground.tensorflow.org/">g</a><u><a href="http://playground.tensorflow.org/">round.tensorflow.or</a></u><a href="http://playground.tensorflow.org/">g)</a><a href="http://playground.tensorflow.org/">.</a>

Starter Code

In the starter code, you are provided with a function that loads data into minibatches for training and testing in PyTorch.

Benchmark

Train for a least 30 epochs to get a least 44mm avg error. The test result(mm error) should be in the following sequence <strong>direct. discuss. eat. greet. phone photo pose purch. sit sitd. somke wait walkd. walk walkT avg</strong>

<strong>Problem 1:</strong>

<a href="http://localhost:8888/nbconvert/html/(https:/arxiv.org/pdf/1705.03098.pdf">{60 points} Let us first start by trying to reproduce the testing accuracy obtained by in the </a><u><a href="http://localhost:8888/nbconvert/html/(https:/arxiv.org/pdf/1705.03098.pdf">paper</a></u>

<u><a href="http://localhost:8888/nbconvert/html/(https:/arxiv.org/pdf/1705.03098.pdf">((https://arxiv.or</a></u><a href="http://localhost:8888/nbconvert/html/(https:/arxiv.org/pdf/1705.03098.pdf">g</a><u><a href="http://localhost:8888/nbconvert/html/(https:/arxiv.org/pdf/1705.03098.pdf">/pdf/1705.03098.pdf)</a></u><a href="http://localhost:8888/nbconvert/html/(https:/arxiv.org/pdf/1705.03098.pdf"> above using PyTorch. The 2D pose of a person is represe</a>nted as a set of 2D coordinates for each of their n = 32 joints i.e P<sub>i</sub><sup>2D</sup>= {(x<sup>1</sup><sub>i </sub>, y<sub>i</sub><sup>1</sup>), . . . , (x<sub>i</sub><sup>32</sup> , y<sub>i</sub><sup>32</sup>)}, where (x<sup>j</sup><sub>i</sub>, y<sub>i</sub><sup>j</sup>) are the 2D coordinates of the j’th joint of the i’th sample. Similarly, the 3D pose of a person is <sup>P</sup><sub>i</sub><sup>3D</sup> = {

(x<sup>1</sup><sub>i </sub>, y<sub>i</sub><sup>1</sup>, z<sub>i</sub><sup>1</sup>), . . . , (x<sup>32</sup><sub>i </sub>, y<sub>i</sub><sup>32</sup>, z<sub>i</sub><sup>32</sup>)}, where (x<sup>j</sup><sub>i</sub>, y<sub>i</sub><sup>j</sup>, z<sub>i</sub><sup>j</sup>) are the 3D coordinates of the j’th joint of the i’th sample.

The only data given to you is the ground truth 3D pose and the 2D pose calculated using the camera parameters. You are going to train a network f<sub>θ </sub>: R<sup>2n </sup>→ R<sup>3n</sup> that takes as input the <sup>P</sup><sub>i</sub><sup>2D</sup> and tries to regress the ground truth 3D pose P<sub>i</sub><sup>3D</sup>. The loss function to train this network would be the L2 loss between the ground truth and the predicted pose

M

L;            for a minibatch of size M           (2)

i=1

Download the Human3.6M Dataset <u><a href="https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip">here </a></u><a href="https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip">(</a><u><a href="https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip">https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip)</a></u><a href="https://www.dropbox.com/s/e35qv3n6zlkouki/h36m.zip">.</a>

<strong>Bonus</strong>: Every 1mm drop in test error from 44mm till 40mm gets you 2 extra points, and every 1mm drop below 40mm gets you 4 extra points.

Report the test result(mm error) in the following sequence <strong>direct. discuss. eat. greet. phone photo pose purch. sit sitd. somke wait walkd. walk walkT avg</strong>

<strong>Problem 2:</strong>

{40 points} In this task, we’re going to tackle the situation of having a faulty 3D sensor. Since the sensor is quite old it’s joint detections are quite noisy:

x^ = x<sub>GT </sub>+ ϵ<sub>x </sub>y^ = y<sub>GT </sub>+ ϵ<sub>y</sub>

z^ = z<sub>GT </sub>+ ϵ<sub>z</sub>

Where, (x<sub>G</sub>T, y<sub>G</sub>T, z<sub>G</sub>T) are the ground truth joint locations, (x^, y^, z^) are the noisy measurements detected by our sensor and (ϵ<sub>x</sub>, ϵ<sub>y</sub>, ϵ<sub>z</sub>) are the noise values. Being grad students, we’d much rather the department spend money for free coffee and doughnuts than on a new 3D sensor. Therefore, you’re going to denoise the noisy data using a linear Kalman filter.

<strong>Modelling the state using velocity and acceleration</strong>: We assume a simple, if unrealistic model, of our system – we’re only going to use the position, velocity and acceleration of the joints to denoise the data. The underlying equations representing our assumptions are:

∂x

xt+1 = xt + ∂tt δt + 0.5 ∗ ∂∂2tx2t δt2     (1)

∂y                        2

yt+1 = yt + ∂tt δt + 0.5 ∗ ∂∂ty2t δt2 (2) z<sub>t+1 </sub>= z<sub>t </sub>+ ∂tt δt + 0.5 <sup>∗ </sup>∂∂2tz<sub>2</sub>t δt2 (3) ∂z

The only measurements/observations we have (i.e our ‘observation space’) are the noisy joint locations as recorded by the 3D sensors o<sub>t </sub>= (x^<sub>t</sub>, y<sup>^</sup><sub>t</sub>, z^<sub>t</sub>). The corresponding state-space would be z<sub>t </sub>= (x<sub>t</sub>, y<sub>t</sub>, z<sub>t</sub>, .

Formallly, a linear Kalman filter assumes the underlying dynamics of the system to be a linear Gaussian model i.e.

z<sub>0 </sub>∼ N(μ<sub>0</sub>, Σ<sub>0</sub>)

z<sub>t</sub>+1 = Az<sub>t </sub>

o<sub>t </sub>= Cz<sub>t </sub>ϵ ϵ

where, A and C are the transition_matrix and observation_matrix respectively, that you are going to define based on equations (1), (2) and (3). The intitial estimates of other parameters can be assumed as:

initial_state_mean := μ<sub>0 </sub>= mean(given data)

initial_state_covariance := Σ<sub>0 </sub>= Cov(given data)

transition_offset := b = 0

observation_offset := d = 0

transition_covariance := Q = I observation_covariance := R = I

The covariance matrices Q and R are hyperparameters that we initalize as identity matrices. In the code below, you must define A and C and use <u><a href="https://pykalman.github.io/">pykalman </a></u><a href="https://pykalman.github.io/">(</a><u><a href="https://pykalman.github.io/">https://p</a></u><a href="https://pykalman.github.io/">y</a><u><a href="https://pykalman.github.io/">kalman.</a></u><a href="https://pykalman.github.io/">g</a><u><a href="https://pykalman.github.io/">ithub.io/)</a></u><a href="https://pykalman.github.io/">,</a> a dedicated library for kalman filtering in python, to filter out the noise in the data.

(<strong>Hint:</strong>  Gradients could be calculated using np.gradient or manually using finite differences  You can assume the frame rate to be 50Hz)

For more detailed resources related to Kalman filtering, please refer to:  <u><a href="http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf">http://web.mit.edu/kirtle</a></u><a href="http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf">y</a><u><a href="http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf">/kirtle</a></u><a href="http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf">y</a><u><a href="http://web.mit.edu/kirtley/kirtley/binlustuff/literature/control/Kalman%20filter.pdf">/binlustuff/literature/control/Kalman%20filter.pdf</a></u>

<u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">(http://web.mit.edu/kirtle</a></u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">y</a><u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">/kirtle</a></u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">y</a><u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">/binlustuff/literature/control/Kalman%20filter.pdf)</a></u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">  </a><u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">https://www.bzar</a></u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">g</a><u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">.com/p/howa-kalman-filter-works-in-pictures/ </a></u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">(</a><u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">https://www.bzar</a></u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">g</a><u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">.com/p/how-a-kalman-filter-works-in-pictures/)</a></u><a href="https://www.bzarg.com/p/how-a-kalman-filter-works-in-pictures/">  </a><u><a href="https://stanford.edu/class/ee363/lectures/kf.pdf">https://stanford.edu/class/ee363/lectures/kf.pdf </a></u><a href="https://stanford.edu/class/ee363/lectures/kf.pdf">(</a><u><a href="https://stanford.edu/class/ee363/lectures/kf.pdf">https://stanford.edu/class/ee363/lectures/kf.pdf)</a></u>