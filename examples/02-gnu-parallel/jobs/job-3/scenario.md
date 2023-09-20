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
0.108
</td>
<td>
0.061
</td>
<td>
0.009
</td>
<td>
0.108
</td>
<td>
0.209
</td>
<td>
4000
</td>
<td>
796.896
</td>
<td>
1017.401
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
kernel_00
</td>
<td>
0.848
</td>
<td>
0.098
</td>
<td>
0.685
</td>
<td>
0.848
</td>
<td>
1.010
</td>
<td>
4000
</td>
<td>
815.286
</td>
<td>
1128.889
</td>
<td>
1.008
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
0.437
</td>
<td>
3.340
</td>
<td>
-5.153
</td>
<td>
0.405
</td>
<td>
5.912
</td>
<td>
4000
</td>
<td>
402.277
</td>
<td>
573.527
</td>
<td>
1.008
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
-2.773
</td>
<td>
2.935
</td>
<td>
-7.823
</td>
<td>
-2.615
</td>
<td>
1.753
</td>
<td>
4000
</td>
<td>
427.243
</td>
<td>
706.961
</td>
<td>
1.009
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
0.146
</td>
<td>
2.834
</td>
<td>
-4.466
</td>
<td>
0.166
</td>
<td>
4.864
</td>
<td>
4000
</td>
<td>
413.283
</td>
<td>
719.967
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
kernel_07
</td>
<td>
3.069
</td>
<td>
2.103
</td>
<td>
-0.293
</td>
<td>
2.985
</td>
<td>
6.662
</td>
<td>
4000
</td>
<td>
350.825
</td>
<td>
747.189
</td>
<td>
1.023
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
-4.979
</td>
<td>
1.770
</td>
<td>
-8.086
</td>
<td>
-4.916
</td>
<td>
-2.208
</td>
<td>
4000
</td>
<td>
480.780
</td>
<td>
821.844
</td>
<td>
1.013
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
-0.413
</td>
<td>
0.817
</td>
<td>
-1.748
</td>
<td>
-0.425
</td>
<td>
0.945
</td>
<td>
4000
</td>
<td>
506.435
</td>
<td>
941.094
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
4.411
</td>
<td>
0.908
</td>
<td>
2.940
</td>
<td>
4.439
</td>
<td>
5.846
</td>
<td>
4000
</td>
<td>
381.151
</td>
<td>
674.231
</td>
<td>
1.015
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
0.632
</td>
<td>
1.546
</td>
<td>
-1.918
</td>
<td>
0.635
</td>
<td>
3.173
</td>
<td>
4000
</td>
<td>
472.937
</td>
<td>
862.807
</td>
<td>
1.007
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
0.331
</td>
<td>
0.644
</td>
<td>
-0.727
</td>
<td>
0.379
</td>
<td>
1.312
</td>
<td>
4000
</td>
<td>
383.370
</td>
<td>
600.256
</td>
<td>
1.015
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
16.865
</td>
<td>
15.382
</td>
<td>
4.741
</td>
<td>
12.792
</td>
<td>
41.386
</td>
<td>
4000
</td>
<td>
685.515
</td>
<td>
1810.112
</td>
<td>
1.010
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
-0.032
</td>
<td>
0.046
</td>
<td>
-0.109
</td>
<td>
-0.032
</td>
<td>
0.046
</td>
<td>
4000
</td>
<td>
983.101
</td>
<td>
2028.071
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
-1.005
</td>
<td>
2.890
</td>
<td>
-5.886
</td>
<td>
-0.861
</td>
<td>
3.457
</td>
<td>
4000
</td>
<td>
414.577
</td>
<td>
562.438
</td>
<td>
1.009
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
1.324
</td>
<td>
2.364
</td>
<td>
-2.428
</td>
<td>
1.220
</td>
<td>
5.245
</td>
<td>
4000
</td>
<td>
476.796
</td>
<td>
907.740
</td>
<td>
1.009
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
0.348
</td>
<td>
2.653
</td>
<td>
-3.806
</td>
<td>
0.281
</td>
<td>
4.979
</td>
<td>
4000
</td>
<td>
403.040
</td>
<td>
788.236
</td>
<td>
1.014
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
2.209
</td>
<td>
1.812
</td>
<td>
-0.643
</td>
<td>
2.104
</td>
<td>
5.301
</td>
<td>
4000
</td>
<td>
387.359
</td>
<td>
856.965
</td>
<td>
1.010
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
-1.060
</td>
<td>
1.744
</td>
<td>
-3.985
</td>
<td>
-1.012
</td>
<td>
1.688
</td>
<td>
4000
</td>
<td>
521.613
</td>
<td>
823.517
</td>
<td>
1.004
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
-5.152
</td>
<td>
0.884
</td>
<td>
-6.616
</td>
<td>
-5.149
</td>
<td>
-3.678
</td>
<td>
4000
</td>
<td>
335.297
</td>
<td>
650.063
</td>
<td>
1.007
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
0.039
</td>
<td>
0.812
</td>
<td>
-1.308
</td>
<td>
0.057
</td>
<td>
1.335
</td>
<td>
4000
</td>
<td>
481.811
</td>
<td>
918.204
</td>
<td>
1.004
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
1.029
</td>
<td>
1.557
</td>
<td>
-1.341
</td>
<td>
0.918
</td>
<td>
3.748
</td>
<td>
4000
</td>
<td>
340.145
</td>
<td>
750.263
</td>
<td>
1.009
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
0.223
</td>
<td>
0.497
</td>
<td>
-0.596
</td>
<td>
0.226
</td>
<td>
1.018
</td>
<td>
4000
</td>
<td>
487.893
</td>
<td>
782.550
</td>
<td>
1.006
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
11.056
</td>
<td>
10.593
</td>
<td>
2.648
</td>
<td>
7.988
</td>
<td>
28.496
</td>
<td>
4000
</td>
<td>
533.252
</td>
<td>
1039.338
</td>
<td>
1.005
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
-0.025
</td>
<td>
0.048
</td>
<td>
-0.104
</td>
<td>
-0.025
</td>
<td>
0.052
</td>
<td>
4000
</td>
<td>
891.469
</td>
<td>
1541.989
</td>
<td>
1.001
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
0.009
</td>
<td>
0.042
</td>
<td>
-0.059
</td>
<td>
0.007
</td>
<td>
0.081
</td>
<td>
4000
</td>
<td>
864.135
</td>
<td>
1589.408
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
kernel_02
</td>
<td>
1.017
</td>
<td>
0.071
</td>
<td>
0.900
</td>
<td>
1.018
</td>
<td>
1.135
</td>
<td>
4000
</td>
<td>
884.299
</td>
<td>
1693.224
</td>
<td>
1.003
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
0.063
</td>
<td>
0.041
</td>
<td>
-0.004
</td>
<td>
0.063
</td>
<td>
0.133
</td>
<td>
4000
</td>
<td>
768.447
</td>
<td>
1298.448
</td>
<td>
1.008
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
0.889
</td>
<td>
0.070
</td>
<td>
0.774
</td>
<td>
0.889
</td>
<td>
1.004
</td>
<td>
4000
</td>
<td>
837.616
</td>
<td>
1547.497
</td>
<td>
1.004
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
