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
    liesel.goose.engine - WARNING - Errors per chain for kernel_00: 1, 0, 0, 0 / 25 transitions
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
-0.120
</td>
<td>
0.062
</td>
<td>
-0.222
</td>
<td>
-0.119
</td>
<td>
-0.019
</td>
<td>
4000
</td>
<td>
838.956
</td>
<td>
1104.162
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
1.167
</td>
<td>
0.105
</td>
<td>
0.993
</td>
<td>
1.167
</td>
<td>
1.340
</td>
<td>
4000
</td>
<td>
781.599
</td>
<td>
1268.438
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
0.070
</td>
<td>
3.038
</td>
<td>
-5.321
</td>
<td>
0.180
</td>
<td>
4.813
</td>
<td>
4000
</td>
<td>
454.978
</td>
<td>
639.792
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
kernel_07
</td>
<td>
-0.298
</td>
<td>
2.355
</td>
<td>
-4.091
</td>
<td>
-0.366
</td>
<td>
3.599
</td>
<td>
4000
</td>
<td>
490.321
</td>
<td>
972.800
</td>
<td>
1.003
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
-0.824
</td>
<td>
2.560
</td>
<td>
-5.187
</td>
<td>
-0.715
</td>
<td>
3.220
</td>
<td>
4000
</td>
<td>
487.258
</td>
<td>
746.471
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
-0.792
</td>
<td>
1.651
</td>
<td>
-3.496
</td>
<td>
-0.817
</td>
<td>
1.993
</td>
<td>
4000
</td>
<td>
403.241
</td>
<td>
754.644
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
-0.943
</td>
<td>
1.705
</td>
<td>
-3.800
</td>
<td>
-0.910
</td>
<td>
1.798
</td>
<td>
4000
</td>
<td>
396.925
</td>
<td>
897.129
</td>
<td>
1.018
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
-0.557
</td>
<td>
0.741
</td>
<td>
-1.802
</td>
<td>
-0.562
</td>
<td>
0.674
</td>
<td>
4000
</td>
<td>
530.226
</td>
<td>
979.303
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
-5.821
</td>
<td>
0.857
</td>
<td>
-7.251
</td>
<td>
-5.785
</td>
<td>
-4.466
</td>
<td>
4000
</td>
<td>
376.023
</td>
<td>
696.262
</td>
<td>
1.013
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
-0.101
</td>
<td>
1.390
</td>
<td>
-2.408
</td>
<td>
-0.072
</td>
<td>
2.083
</td>
<td>
4000
</td>
<td>
465.222
</td>
<td>
906.459
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
kernel_07
</td>
<td>
-1.453
</td>
<td>
0.560
</td>
<td>
-2.376
</td>
<td>
-1.420
</td>
<td>
-0.596
</td>
<td>
4000
</td>
<td>
402.717
</td>
<td>
677.035
</td>
<td>
1.013
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
11.119
</td>
<td>
10.111
</td>
<td>
3.157
</td>
<td>
8.467
</td>
<td>
27.053
</td>
<td>
4000
</td>
<td>
796.529
</td>
<td>
2268.923
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
0.069
</td>
<td>
0.043
</td>
<td>
-0.001
</td>
<td>
0.069
</td>
<td>
0.141
</td>
<td>
4000
</td>
<td>
930.640
</td>
<td>
1661.786
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
-0.402
</td>
<td>
2.886
</td>
<td>
-5.198
</td>
<td>
-0.236
</td>
<td>
4.004
</td>
<td>
4000
</td>
<td>
465.855
</td>
<td>
741.290
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
kernel_04
</td>
<td>
-0.132
</td>
<td>
2.292
</td>
<td>
-3.944
</td>
<td>
-0.094
</td>
<td>
3.525
</td>
<td>
4000
</td>
<td>
476.631
</td>
<td>
867.941
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
kernel_04
</td>
<td>
-0.796
</td>
<td>
2.763
</td>
<td>
-5.386
</td>
<td>
-0.731
</td>
<td>
3.549
</td>
<td>
4000
</td>
<td>
342.524
</td>
<td>
500.720
</td>
<td>
1.016
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
2.777
</td>
<td>
1.682
</td>
<td>
0.149
</td>
<td>
2.702
</td>
<td>
5.519
</td>
<td>
4000
</td>
<td>
385.165
</td>
<td>
820.549
</td>
<td>
1.008
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
1.884
</td>
<td>
1.702
</td>
<td>
-0.872
</td>
<td>
1.850
</td>
<td>
4.719
</td>
<td>
4000
</td>
<td>
524.622
</td>
<td>
1263.476
</td>
<td>
1.009
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
5.085
</td>
<td>
0.803
</td>
<td>
3.760
</td>
<td>
5.080
</td>
<td>
6.478
</td>
<td>
4000
</td>
<td>
378.536
</td>
<td>
929.540
</td>
<td>
1.008
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
0.785
</td>
<td>
0.804
</td>
<td>
-0.507
</td>
<td>
0.787
</td>
<td>
2.112
</td>
<td>
4000
</td>
<td>
445.056
</td>
<td>
760.327
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
0.757
</td>
<td>
1.420
</td>
<td>
-1.424
</td>
<td>
0.679
</td>
<td>
3.197
</td>
<td>
4000
</td>
<td>
432.970
</td>
<td>
915.915
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
kernel_04
</td>
<td>
0.580
</td>
<td>
0.532
</td>
<td>
-0.242
</td>
<td>
0.551
</td>
<td>
1.489
</td>
<td>
4000
</td>
<td>
395.193
</td>
<td>
621.982
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
11.372
</td>
<td>
10.181
</td>
<td>
2.887
</td>
<td>
8.471
</td>
<td>
29.026
</td>
<td>
4000
</td>
<td>
553.920
</td>
<td>
1299.327
</td>
<td>
1.004
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
0.045
</td>
<td>
0.046
</td>
<td>
-0.031
</td>
<td>
0.045
</td>
<td>
0.118
</td>
<td>
4000
</td>
<td>
744.904
</td>
<td>
912.223
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
-0.014
</td>
<td>
0.043
</td>
<td>
-0.085
</td>
<td>
-0.016
</td>
<td>
0.057
</td>
<td>
4000
</td>
<td>
897.505
</td>
<td>
1723.877
</td>
<td>
1.003
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
0.951
</td>
<td>
0.075
</td>
<td>
0.827
</td>
<td>
0.950
</td>
<td>
1.075
</td>
<td>
4000
</td>
<td>
894.153
</td>
<td>
1621.784
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
-0.035
</td>
<td>
0.042
</td>
<td>
-0.102
</td>
<td>
-0.035
</td>
<td>
0.035
</td>
<td>
4000
</td>
<td>
756.809
</td>
<td>
1207.443
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
1.042
</td>
<td>
0.070
</td>
<td>
0.926
</td>
<td>
1.042
</td>
<td>
1.157
</td>
<td>
4000
</td>
<td>
830.183
</td>
<td>
1446.170
</td>
<td>
1.003
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
2
</td>
<td>
0.001
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
