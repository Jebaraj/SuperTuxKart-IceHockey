# SuperTuxKart Ice-Hockey AI UT Summer 2021 Deep Learning Final Project

This was a project done for UT Austin's Deep Learning Course in Summer 2021. We discuss the approach we took for this project in this write up but have ommited the code to preserve the integrity of the assignment.


#### Team Members: Jebaraj Vasudevan, Vishank Bhatia, Mohammed Belcaid, Subramanian Aiyer 


![Our Agents against the AI (highest difficulty)](media/Best_Performing_Agent.gif)  

*Gameplay showing ball chasing, positioning, and reversal when we pass the puck. Red is prediction, Green is truth*

## Introduction

In this project, we built a SuperTuxKart ice-hockey agent based on a supervised learning approach. In this game, 2 agents from our team will play against two rival agents from an opponent’s team or the AI. To win the game, our agents have to score more goals than the opponents within 2 minutes. We tried using two approaches to solve this task. The first approach was Dagger-based imitation learning and the second approach was a supervised learning based vision and control method. Finally, we decided to implement the vision-based controller due to its superior behaviour compared to the other approach in the given time-constraint. Using the vision-based controller approach we were able to train an efficient deep network (inference at less than 100ms / step in a CPU) with an average score 2-0 against the game AI across all difficulty levels.

## Strategies Explored

We decided to try two approaches to implement our agents: a Dagger-based imitation controller and a vision-based controller. In the imitation approach, we collected 4800 images and the corresponding action pairs of the AI set to difficulty level 2 which is the highest difficulty level. A CNN model similar to HW3 was used with 6 outputs. MSELoss with BCEwithLogitsLoss were used to train this classification network. MSELoss was used for continuous outputs like steer and acceleration while BCEwithLogitsLoss was used for other discrete actions. While imitation learning was quite easy to implement, the Dagger part where we need the AI actions to supervise the agent seemed difficult and time-consuming to implement. In the end, we decided on a vision-based controller approach where we detect the puck and steer the agent accordingly. Specifically, we trained a U-net based segmentation network to detect the puck on the image, compute the angle between the kart and the puck, rotate by that angle and go to the puck, then rotate again and push the puck towards the goal. The data collection and methodology is explained in the sections below.

## Approach

In order to convert from 3D kart coordinates to 2D coordinates
we omitted the Y-direction since it gives only the
vertical coordinate of the puck while the X-Z plane defines
the playing field. We used the center of the goal co-ordinates
as global constants and chose the “wilbert” cart, because from
the image perspective of the player, the model doesn’t get
confused between the puck and the kart as it has a similar
darker color of the puck.
At the start of the game, we initialize status variables which
keep track of each function call to the agent, track of the puck
location change, and the rescue variables which help rescue
the cart when it gets stuck (e.g. stuck to the field’s wall). We
get the kart’s position and velocity from player-info.kart, and
use the kart-front vector that defines the orientation of the kart.
To locate the puck, we use the detect() method that predicts
the heat map of the puck in the image, the predicted location
of the puck in the image, and the list of the peaks of the
heatmap of the image along with its scores.
### A. Strategy to check for puck visibility
In order to know if the puck is visible in the image, we filter
the list of [score, pixel-x, pixel-y] based on the highest score,
so as to filter out the false positives detected by the model
and only focus on highest confidence detections. Then, we
calculate the mean of these scores, compare it with a threshold
and also calculate the sum of the thresholded heatmap of
the puck and compare it with the threshold to determine
puck visibility. We use both the scores and the sum of the
thresholded heatmap to ensure that we avoid false positive
detections in the scores while taking into account only strong
signals in the heatmap to detect the presence of the puck.
### B. Strategy to estimate puck size
In order to estimate the size of the puck in the image,
we threshold the predicted heatmap-mask to sum up the
values with higher confidence, then we normalize by applying
torch.sum on the heatmap-mask to have a value in [0,1]. This
allows us to estimate the visual size of the puck in the image,
i.e. how close or away is the puck from the kart.
### C. Strategy to orient the puck towards opposition goal
If the puck is visible in the image, we use the predicted xcoordinate
of the normalized puck location to know if it lies to
the left or the right side of the opposition goal. If the puck is
not visible and we use the last known puck location for a few
steps and if the puck is not visible for quite some time, we
reverse back to our goal location. Once the puck is in sight,
we compute the angle between the kart’s direction and the
puck. The position of the kart w.r.t to the goal is calculated in a similar
way using the tangent of the angle between the kart orientation
vector and the kart to opposition goal vector to determine if
the kart is to the left or right of the goal depending on the
sign of the arctan2().
If the kart is within a particular angle on either side of the
opposition goal, depending on how far the opposition goal is,
an importance metric is calculated which gives more weight
to the relative position of the puck compared to the opposition
goal angle and how close the kart is to the opposition goal. If
not, the kart just goes towards the puck and neglects the angle
to the opposition goal.
### D. Strategy to dribble the puck
If the puck is visible, we use the puck size to determine
acceleration, steer and nitro values. If the puck is closer
to the kart, lesser steer and acceleration is applied with no
nitro to gently dribble the puck, while if the puck is farther
away, higher steer, with higher acceleration along with nitro
is applied to get to the puck as quickly as possible
### E. Strategy to recover the karts
In order to recover the kart from getting stuck against a wall,
two queues were used to keep track of the last few frames of
the kart velocity and if the average of the last few frames is
below a certain threshold, then the recovery routine kicks in.
The routine for the next few frames reverses the kart towards
our own goal and helps the kart to get out of any obstacles.
### F. Other strategies tried but not implemented
[1] We tried to implement a routine to always face towards
the opposition goal, once the kart is no longer facing the
opposition goal. But its drawback was that once we try
to re-orient the kart towards the opposite goal, we end up
losing sight of the puck
[2] We also tried to implement a goalkeeper approach initially
by having one kart play a defensive role while the
other kart plays an offensive role. Though this helped to
win or draw against the AI, it resulted in low scoring
games as one kart is always tied down closer to our
own goal. Since the grading heavily weighs high scores
compared to draws, we decided to make both players
offensive to increase the chances of our team scoring
against the AI and other teams
### G. Limitations of our controller
Though our controller was consistently winning against the
AI in all levels both as team 0 and as team 1, it has some
limitations.
[1] Specifically, once the controller is out of phase with
the opposite goal, it no longer takes into account the
opposition goal position and may end up scoring against
our own goal instead of the opposite goal
[2] Also, when both controllers in our team lose sight of
the puck, they both fall back to our own goal and keep
waiting for the puck to appear in their sight, which
sometimes does not work if the AI player also stops
playing actively
[3] The controller performed differently in different platforms
due to variation though it was able to consistently
win or draw against the AI in almost all games.

## Conclusion

Our vision-based controller strategy of following the puck
performed remarkably well vs the Game AI. This strategy
of continuously following the puck and keeping it moving
towards the opposition goal ensured that our agents always had
the puck under their control. We also faced implementation
challenges for the vision-based controller approach like handcoding
the strategy to score goals. We had to account for
multiple scenarios including when the kart gets stuck to the
field walls and needs to be rescued, or when the puck is not
being seen by both the agents. We tried to overcome these
challenges but enumerating infinite scenarios as a strategy
seems to be very difficult, so we tried to come up with
a strategy that is generalist enough to ensure the controller
performs generally well. For future work, we could develop a
reinforcement learning agent that uses the tuned controller as
an oracle with reward defined as the number of goals scored.
This approach would help to eliminate the manual fine tuning
required for the vision-based controller to work.

References

[1] https://arxiv.org/pdf/1505.04597.pdf

[2] https://ieeexplore.ieee.org/document/4803973

[3] https://arxiv.org/pdf/2103.14823.pdf