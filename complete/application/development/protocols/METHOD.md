# Bounded spectral updates: methodological development

This document records the revised target before assessment of the twelve new
datasets. The earlier six-dataset study remains exploratory evidence and must
remain visible in the supplement, including its unfavorable results.

## Target and guarantee

A frozen classifier defines correctness C. A reference reliability score r is
chosen on calibration data from output, time, representation, agreement,
atypicality, combined-control, raw-score, and intercept alternatives. A candidate
q and its log-odds budget B are also selected on calibration data. Deployment
uses q only after a separate gate certifies positive expected paired log-score
gain relative to r. This is a **proper-score improvement contract**. It is not
a ranking, selective-risk, calibration-error, conditional-independence, or
individual-correctness certificate. Those other endpoints are separately
reported on assessment data.

The revised contract differs from the original multi-endpoint gate. It is not
valid to present the change as a more powerful implementation of the same
guarantee. The paper must explicitly state the narrower guarantee.

## Bounded adjustment

For a fitted unconstrained score s, set

q = sigmoid(logit(r) + clip(logit(s)-logit(r), -B, B)).

Writing a=logit(r), b=logit(q)-a, the realized log-score gain is

d = C b - log(1+exp(a+b)) + log(1+exp(a)).

The derivative with respect to b is C-sigmoid(a+b), in [-1,1]. Therefore
|d| <= |b| <= B for both labels and every example. Probability clipping to
[epsilon,1-epsilon], with r already in that interval, can only contract the
log-odds adjustment, so the same bound remains valid. In particular the
score-risk downside is bounded for the candidate even before authorization.
The feature invariances alone do not imply invariance of q, because r and the
output features can change under transformations.

## Gate

Condition on all calibration choices and training data. Gate rows must be iid
from the claimed target population and independent of those choices. Under the
null E[d]<=0, for any predictable lambda_t in [0,1/B],

K_n = product_t (1+lambda_t d_t)

is a nonnegative supermartingale starting at one: each factor is nonnegative,
and its conditional expectation is at most one. Use an equal-weight mixture
of nine constant fractions lambda B = 2^(-8),...,1 and one predictable bet

lambda_t = clip(S_(t-1) / (2 V_(t-1)+B^2), 0, 1/(2B)),

where S is the previous sum of gains and V is the previous sum of squared
gains. The current label never enters its own bet. Mixtures retain validity.
Across M=48 frozen configurations, authorize at K>=M/alpha, alpha=.05.
Markov's inequality and a union bound yield familywise probability at most
alpha of authorizing any nonpositive-gain candidate. Dependence across
configurations, including shared examples and related backbones, is allowed.
Ville's inequality supplies the analogous guarantee for repeated monitoring
of the same fixed candidates. The reported experiment uses one fixed gate
size, rather than exploiting optional stopping.

This construction specializes established betting inference (Waudby-Smith and
Ramdas, JRSS B 2024). The paper must not claim to invent test martingales,
bounded-mean testing, or multiple-testing control.

## Consistency of the adaptive component

For iid bounded gains with mean mu>0 and second moment v>0, the strong law
gives lambda_t -> lambda*=min(mu/(2v),1/(2B))>0. The map log(1+lambda d)
is uniformly 2B-Lipschitz for lambda in [0,1/(2B)], so replacing lambda_t
with lambda* changes average log wealth by a term tending to zero. The strong
law and log(1+x)>=x-x^2 on [-1/2,1/2] give

lim n^(-1) log K_n = E log(1+lambda* d)
                  >= lambda* mu - (lambda*)^2 v
                  >= lambda* mu/2 > 0.

Consequently the adaptive component, and hence its fixed positive-weight
mixture, eventually crosses any fixed authorization threshold almost surely.
This is an asymptotic statement, not a small-sample power promise. Adding the
adaptive component avoids the minimum detectable mean implicit in a finite
grid of strictly positive constant bets.

## Information boundary

Positive fitted gain remains distinct from positive conditional information.
The exact identity Delta_fit = I(C;G|H)+A0-A1 is retained. The gate can
legitimately authorize a richer model under conditional independence if it
reduces approximation error. Conditional-null simulations must distinguish
this from falsely authorizing a candidate with nonpositive fitted gain.

## Study design

Official TRAIN fits the frozen classifiers. Official TEST is an audit pool:
40% calibration, 30% gate, 30% assessment, assigned by a label-independent
SHA-256 ordering. Calling the entire official TEST untouched would be false.
Neither assessment outcomes nor gate outcomes select learners, features,
budgets, backbones, or datasets. All twelve declared datasets are reported.
Failures of computation or missing classes are recorded, not silently dropped.
The new suite includes two multivariate datasets. MiniRocket and Hydra use
published transformations at their standard feature settings and ridge heads.
Their softmax-transformed decision scores are not assumed calibrated.

The 48 configurations are twelve datasets x two backbones x two seeds. They
are not 48 independent statistical replicates. Overall effects are macro
averages over datasets after averaging the four configurations within each.
Uncertainty for benchmark generalization uses dataset-level paired resampling
and is descriptive for this selected suite. Gate validity has separate iid
sampling assumptions which benchmark files cannot establish by themselves.
