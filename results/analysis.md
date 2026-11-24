# Analysis

## 6.1 Person-independent validation

Validation subjects 3, 15, 25, and 30 are held out before training. Random window-level splitting would leak highly similar, 50%-overlapping windows across adjacent segments and could let the network recognize a person rather than general activity. Subject-level separation gives an honest estimate of generalization.

## 6.2 Most confused activities

The largest off-diagonal confusion counts are STANDING -> SITTING (103 windows) and SITTING -> STANDING (75 windows). In the waist-mounted recordings, SITTING and STANDING have weak body motion and similar posture-related acceleration/gyroscope patterns, so they are difficult to separate. Their example-waveforms also show much less dynamic change than walking activities.

## 6.3 Why LAYING is easy

LAYING is strongly separated by the gravity component, especially the total-acceleration vertical channel and the body-acceleration response. The phone orientation is nearly stable in lying posture, producing a consistent acceleration direction that differs from upright and moving postures. In this run, LAYING per-class accuracy is 0.9497.

## 6.4 Attention peakiness

The mean maximum attention weight is 0.312863 for moving activities and 0.019103 for still activities. These values are estimates of how concentrated each prediction is. A larger value means the network concentrates evidence in fewer moments; a smaller value means it uses more of the 128-step window. Useful temporal information can occur over several cycles for movement, while stationary activities may depend on persistent orientation information.

## 6.5 Why bidirectional processing is allowed

The test window is a complete offline 2.56-second segment, so every point is available when the network makes a decision. Bidirectional processing is therefore appropriate for this activity-recognition task. It would not be appropriate for a strictly real-time or causal stream (for example, live fall detection), where future sensor samples would not yet exist.

## 6.6 RNN versus LSTM and clipping

A plain RNN repeatedly rewrites one hidden state with a tanh transformation, so repeated multiplication can cause gradients to vanish or explode over 128 steps. The LSTM additionally maintains a cell state:

`c_t = f_t * c_(t-1) + i_t * g_t`

The forget gate controls retention, while the input gate controls how much of each candidate is added. This additive cell update carries information across many steps more reliably than repeatedly transforming the hidden state alone. Gradient clipping guards against large or exploding parameter-gradient updates; it does not prevent all vanishing gradients, but limits the total gradient norm to 1.0.
