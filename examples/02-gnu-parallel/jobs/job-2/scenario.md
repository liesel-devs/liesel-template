# liesel-template: GNU Parallel example: Scenario

This small simulation study builds on the Quarto example (the
semi-parametric distributional regression model with the bivariate
normal response variable) but uses a parameterized Quarto document to
generate data with different PRNG seeds. GNU Parallel passes the job
number to Quarto as a parameter, which is then used to set the PRNG seed
before generating the data. Finally, the results are stored in an RDS
file for each job.

## Generate some data

Generate five covariates with non-linear effects on the marginal means
and linear effects on the marginal standard deviations and the
correlation parameter of a bivariate normal response variable.

``` r
set.seed(params$job)

n <- 1000

x1 <- runif(n)
x2 <- runif(n)
x3 <- runif(n)
x4 <- runif(n)
x5 <- runif(n)

y <- vapply(1:n, function(i) {
  loc1 <- sin(2 * pi * x1[i])
  loc2 <- cos(2 * pi * x2[i])
  scale1 <- exp(x3[i])
  scale2 <- exp(x4[i])
  cor <- tanh(x5[i])

  mu <- c(loc1, loc2)
  cov <- scale1 * scale2 * cor
  sigma <- matrix(c(scale1^2, cov, cov, scale2^2), nrow = 2)
  mvrnorm(1, mu, sigma)
}, FUN.VALUE = c(0, 0))

y <- t(y)
```

## Configure semi-parametric distributional regression model

Configure the correct data-generating model using RLiesel.

``` r
model <- liesel(
  response = y,
  distribution = py$BivariateNormal,
  predictors = list(
    loc1 = predictor(~s(x1), inverse_link = "Identity"),
    loc2 = predictor(~s(x2), inverse_link = "Identity"),
    scale1 = predictor(~x3, inverse_link = "Exp"),
    scale2 = predictor(~x4, inverse_link = "Exp"),
    cor = predictor(~x5, inverse_link = "Tanh")
  )
)
```

    Installed Liesel version 0.2.4 is compatible, continuing to set up model

## Run MCMC sampler

This MCMC sampler uses IWLS kernels for the regression coefficients and
Gibbs kernels for the smoothing parameters.

``` python
builder = lsl.dist_reg_mcmc(r.model, seed=1337, num_chains=4)
builder.set_duration(warmup_duration=1000, posterior_duration=1000)

engine = builder.build()
```

    liesel.goose.engine - INFO - Initializing kernels...
    liesel.goose.engine - INFO - Done

``` python
engine.sample_all_epochs()
```

    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 75 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 25 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 100 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 200 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_01: 0, 0, 1, 0 / 200 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: SLOW_ADAPTATION, 500 transitions, 25 jitted together
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 0, 1, 0, 0 / 500 transitions
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Starting epoch: FAST_ADAPTATION, 50 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch
    liesel.goose.engine - INFO - Finished warmup
    liesel.goose.engine - INFO - Starting epoch: POSTERIOR, 1000 transitions, 25 jitted together
    liesel.goose.engine - INFO - Finished epoch

``` python

results = engine.get_results()
summary = gs.Summary(results)
```

``` python
summary
```

<p>
<strong>Parameter summary:</strong>
</p>
<table border="0" class="dataframe">
<thead>
<tr style="text-align: right;">
<th>
</th>
<th>
</th>
<th>
kernel
</th>
<th>
mean
</th>
<th>
sd
</th>
<th>
q_0.05
</th>
<th>
q_0.5
</th>
<th>
q_0.95
</th>
<th>
sample_size
</th>
<th>
ess_bulk
</th>
<th>
ess_tail
</th>
<th>
rhat
</th>
</tr>
<tr>
<th>
parameter
</th>
<th>
index
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="2" valign="top">
cor_p0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_00
</td>
<td>
-0.082
</td>
<td>
0.061
</td>
<td>
-0.181
</td>
<td>
-0.082
</td>
<td>
0.018
</td>
<td>
4000
</td>
<td>
755.481
</td>
<td>
1058.967
</td>
<td>
1.007
</td>
</tr>
<tr>
<th>
(1,)
</th>
<td>
kernel_00
</td>
<td>
1.123
</td>
<td>
0.100
</td>
<td>
0.958
</td>
<td>
1.121
</td>
<td>
1.287
</td>
<td>
4000
</td>
<td>
694.626
</td>
<td>
1063.818
</td>
<td>
1.010
</td>
</tr>
<tr>
<th rowspan="9" valign="top">
loc1_np0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_07
</td>
<td>
2.275
</td>
<td>
3.081
</td>
<td>
-2.193
</td>
<td>
2.085
</td>
<td>
7.670
</td>
<td>
4000
</td>
<td>
505.771
</td>
<td>
721.402
</td>
<td>
1.005
</td>
</tr>
<tr>
<th>
(1,)
</th>
<td>
kernel_07
</td>
<td>
1.700
</td>
<td>
2.553
</td>
<td>
-2.309
</td>
<td>
1.588
</td>
<td>
6.085
</td>
<td>
4000
</td>
<td>
521.627
</td>
<td>
904.267
</td>
<td>
1.008
</td>
</tr>
<tr>
<th>
(2,)
</th>
<td>
kernel_07
</td>
<td>
-2.657
</td>
<td>
2.776
</td>
<td>
-7.704
</td>
<td>
-2.453
</td>
<td>
1.546
</td>
<td>
4000
</td>
<td>
459.269
</td>
<td>
861.554
</td>
<td>
1.015
</td>
</tr>
<tr>
<th>
(3,)
</th>
<td>
kernel_07
</td>
<td>
0.064
</td>
<td>
1.574
</td>
<td>
-2.481
</td>
<td>
0.027
</td>
<td>
2.702
</td>
<td>
4000
</td>
<td>
452.326
</td>
<td>
909.924
</td>
<td>
1.009
</td>
</tr>
<tr>
<th>
(4,)
</th>
<td>
kernel_07
</td>
<td>
1.147
</td>
<td>
1.739
</td>
<td>
-1.737
</td>
<td>
1.141
</td>
<td>
3.978
</td>
<td>
4000
</td>
<td>
511.836
</td>
<td>
1028.707
</td>
<td>
1.010
</td>
</tr>
<tr>
<th>
(5,)
</th>
<td>
kernel_07
</td>
<td>
-0.304
</td>
<td>
0.715
</td>
<td>
-1.494
</td>
<td>
-0.308
</td>
<td>
0.882
</td>
<td>
4000
</td>
<td>
605.024
</td>
<td>
1139.932
</td>
<td>
1.001
</td>
</tr>
<tr>
<th>
(6,)
</th>
<td>
kernel_07
</td>
<td>
-5.065
</td>
<td>
0.806
</td>
<td>
-6.405
</td>
<td>
-5.045
</td>
<td>
-3.772
</td>
<td>
4000
</td>
<td>
459.409
</td>
<td>
906.548
</td>
<td>
1.012
</td>
</tr>
<tr>
<th>
(7,)
</th>
<td>
kernel_07
</td>
<td>
0.976
</td>
<td>
1.353
</td>
<td>
-1.279
</td>
<td>
0.977
</td>
<td>
3.179
</td>
<td>
4000
</td>
<td>
546.377
</td>
<td>
958.551
</td>
<td>
1.006
</td>
</tr>
<tr>
<th>
(8,)
</th>
<td>
kernel_07
</td>
<td>
-1.287
</td>
<td>
0.570
</td>
<td>
-2.200
</td>
<td>
-1.283
</td>
<td>
-0.362
</td>
<td>
4000
</td>
<td>
480.385
</td>
<td>
870.948
</td>
<td>
1.011
</td>
</tr>
<tr>
<th>
loc1_np0_tau2
</th>
<th>
()
</th>
<td>
kernel_06
</td>
<td>
12.554
</td>
<td>
12.262
</td>
<td>
3.143
</td>
<td>
9.155
</td>
<td>
33.022
</td>
<td>
4000
</td>
<td>
687.964
</td>
<td>
1564.826
</td>
<td>
1.005
</td>
</tr>
<tr>
<th>
loc1_p0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_08
</td>
<td>
-0.062
</td>
<td>
0.044
</td>
<td>
-0.134
</td>
<td>
-0.061
</td>
<td>
0.013
</td>
<td>
4000
</td>
<td>
953.027
</td>
<td>
1910.700
</td>
<td>
1.001
</td>
</tr>
<tr>
<th rowspan="9" valign="top">
loc2_np0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_04
</td>
<td>
1.228
</td>
<td>
2.887
</td>
<td>
-3.161
</td>
<td>
1.015
</td>
<td>
6.275
</td>
<td>
4000
</td>
<td>
349.167
</td>
<td>
473.868
</td>
<td>
1.020
</td>
</tr>
<tr>
<th>
(1,)
</th>
<td>
kernel_04
</td>
<td>
-0.502
</td>
<td>
2.244
</td>
<td>
-4.275
</td>
<td>
-0.387
</td>
<td>
3.017
</td>
<td>
4000
</td>
<td>
483.885
</td>
<td>
628.402
</td>
<td>
1.004
</td>
</tr>
<tr>
<th>
(2,)
</th>
<td>
kernel_04
</td>
<td>
1.348
</td>
<td>
2.499
</td>
<td>
-2.460
</td>
<td>
1.201
</td>
<td>
5.680
</td>
<td>
4000
</td>
<td>
429.967
</td>
<td>
975.156
</td>
<td>
1.009
</td>
</tr>
<tr>
<th>
(3,)
</th>
<td>
kernel_04
</td>
<td>
-0.451
</td>
<td>
1.429
</td>
<td>
-2.771
</td>
<td>
-0.441
</td>
<td>
1.972
</td>
<td>
4000
</td>
<td>
565.938
</td>
<td>
1036.001
</td>
<td>
1.007
</td>
</tr>
<tr>
<th>
(4,)
</th>
<td>
kernel_04
</td>
<td>
-0.610
</td>
<td>
1.745
</td>
<td>
-3.435
</td>
<td>
-0.619
</td>
<td>
2.246
</td>
<td>
4000
</td>
<td>
431.546
</td>
<td>
830.685
</td>
<td>
1.003
</td>
</tr>
<tr>
<th>
(5,)
</th>
<td>
kernel_04
</td>
<td>
5.550
</td>
<td>
0.783
</td>
<td>
4.254
</td>
<td>
5.542
</td>
<td>
6.880
</td>
<td>
4000
</td>
<td>
363.773
</td>
<td>
811.154
</td>
<td>
1.012
</td>
</tr>
<tr>
<th>
(6,)
</th>
<td>
kernel_04
</td>
<td>
-0.327
</td>
<td>
0.756
</td>
<td>
-1.571
</td>
<td>
-0.305
</td>
<td>
0.884
</td>
<td>
4000
</td>
<td>
460.523
</td>
<td>
776.365
</td>
<td>
1.005
</td>
</tr>
<tr>
<th>
(7,)
</th>
<td>
kernel_04
</td>
<td>
0.840
</td>
<td>
1.276
</td>
<td>
-1.169
</td>
<td>
0.822
</td>
<td>
3.012
</td>
<td>
4000
</td>
<td>
435.378
</td>
<td>
1037.042
</td>
<td>
1.008
</td>
</tr>
<tr>
<th>
(8,)
</th>
<td>
kernel_04
</td>
<td>
-0.078
</td>
<td>
0.505
</td>
<td>
-0.911
</td>
<td>
-0.067
</td>
<td>
0.711
</td>
<td>
4000
</td>
<td>
444.252
</td>
<td>
790.677
</td>
<td>
1.007
</td>
</tr>
<tr>
<th>
loc2_np0_tau2
</th>
<th>
()
</th>
<td>
kernel_03
</td>
<td>
10.341
</td>
<td>
9.105
</td>
<td>
2.911
</td>
<td>
7.507
</td>
<td>
26.976
</td>
<td>
4000
</td>
<td>
680.095
</td>
<td>
1450.746
</td>
<td>
1.008
</td>
</tr>
<tr>
<th>
loc2_p0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_05
</td>
<td>
-0.080
</td>
<td>
0.046
</td>
<td>
-0.157
</td>
<td>
-0.080
</td>
<td>
-0.005
</td>
<td>
4000
</td>
<td>
710.248
</td>
<td>
908.766
</td>
<td>
1.002
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
scale1_p0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_02
</td>
<td>
-0.050
</td>
<td>
0.045
</td>
<td>
-0.121
</td>
<td>
-0.051
</td>
<td>
0.026
</td>
<td>
4000
</td>
<td>
822.365
</td>
<td>
1664.088
</td>
<td>
1.006
</td>
</tr>
<tr>
<th>
(1,)
</th>
<td>
kernel_02
</td>
<td>
1.102
</td>
<td>
0.078
</td>
<td>
0.974
</td>
<td>
1.102
</td>
<td>
1.231
</td>
<td>
4000
</td>
<td>
799.317
</td>
<td>
1502.887
</td>
<td>
1.006
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
scale2_p0_beta
</th>
<th>
(0,)
</th>
<td>
kernel_01
</td>
<td>
0.008
</td>
<td>
0.041
</td>
<td>
-0.057
</td>
<td>
0.007
</td>
<td>
0.076
</td>
<td>
4000
</td>
<td>
805.047
</td>
<td>
1274.679
</td>
<td>
1.007
</td>
</tr>
<tr>
<th>
(1,)
</th>
<td>
kernel_01
</td>
<td>
0.950
</td>
<td>
0.070
</td>
<td>
0.835
</td>
<td>
0.950
</td>
<td>
1.063
</td>
<td>
4000
</td>
<td>
868.083
</td>
<td>
1571.041
</td>
<td>
1.002
</td>
</tr>
</tbody>
</table>
<p>
<strong>Error summary:</strong>
</p>
<table border="0" class="dataframe">
<thead>
<tr style="text-align: right;">
<th>
</th>
<th>
</th>
<th>
</th>
<th>
</th>
<th>
count
</th>
<th>
relative
</th>
</tr>
<tr>
<th>
kernel
</th>
<th>
error_code
</th>
<th>
error_msg
</th>
<th>
phase
</th>
<th>
</th>
<th>
</th>
</tr>
</thead>
<tbody>
<tr>
<th rowspan="2" valign="top">
kernel_00
</th>
<th rowspan="2" valign="top">
90
</th>
<th rowspan="2" valign="top">
nan acceptance prob
</th>
<th>
warmup
</th>
<td>
1
</td>
<td>
0.000
</td>
</tr>
<tr>
<th>
posterior
</th>
<td>
0
</td>
<td>
0.000
</td>
</tr>
<tr>
<th rowspan="2" valign="top">
kernel_01
</th>
<th rowspan="2" valign="top">
90
</th>
<th rowspan="2" valign="top">
nan acceptance prob
</th>
<th>
warmup
</th>
<td>
1
</td>
<td>
0.000
</td>
</tr>
<tr>
<th>
posterior
</th>
<td>
0
</td>
<td>
0.000
</td>
</tr>
</tbody>
</table>

## Visualize estimated splines

Compute estimated functions in Python…

``` python
x = r.model.vars["loc1_np0_X"].value
beta = summary.quantities["mean"]["loc1_np0_beta"]
loc1_hat = np.asarray(x @ beta)

x = r.model.vars["loc2_np0_X"].value
beta = summary.quantities["mean"]["loc2_np0_beta"]
loc2_hat = np.asarray(x @ beta)
```

… and store them in an RDS file.

``` r
df <- data.frame(
  x = c(x1, x2),
  f = c(py$loc1_hat, py$loc2_hat),
  param = rep(c("loc1", "loc2"), each = n),
  job = params$job
)

saveRDS(df, "splines.rds")
```
