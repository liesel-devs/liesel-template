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
0.036
</td>
<td>
0.062
</td>
<td>
-0.066
</td>
<td>
0.036
</td>
<td>
0.137
</td>
<td>
4000
</td>
<td>
760.838
</td>
<td>
1071.007
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
kernel_00
</td>
<td>
0.907
</td>
<td>
0.101
</td>
<td>
0.741
</td>
<td>
0.907
</td>
<td>
1.072
</td>
<td>
4000
</td>
<td>
743.346
</td>
<td>
1249.139
</td>
<td>
1.009
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
5.526
</td>
<td>
4.424
</td>
<td>
-0.762
</td>
<td>
5.071
</td>
<td>
13.433
</td>
<td>
4000
</td>
<td>
400.808
</td>
<td>
643.886
</td>
<td>
1.004
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
2.392
</td>
<td>
3.256
</td>
<td>
-2.814
</td>
<td>
2.253
</td>
<td>
7.924
</td>
<td>
4000
</td>
<td>
460.519
</td>
<td>
832.722
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
1.708
</td>
<td>
3.308
</td>
<td>
-3.616
</td>
<td>
1.656
</td>
<td>
7.507
</td>
<td>
4000
</td>
<td>
401.009
</td>
<td>
744.613
</td>
<td>
1.011
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
-1.101
</td>
<td>
1.895
</td>
<td>
-4.230
</td>
<td>
-1.098
</td>
<td>
2.027
</td>
<td>
4000
</td>
<td>
390.507
</td>
<td>
687.263
</td>
<td>
1.015
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
2.122
</td>
<td>
2.098
</td>
<td>
-1.401
</td>
<td>
2.169
</td>
<td>
5.447
</td>
<td>
4000
</td>
<td>
446.055
</td>
<td>
1000.173
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
kernel_07
</td>
<td>
-0.748
</td>
<td>
0.856
</td>
<td>
-2.150
</td>
<td>
-0.763
</td>
<td>
0.685
</td>
<td>
4000
</td>
<td>
542.028
</td>
<td>
1059.410
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
-7.373
</td>
<td>
0.974
</td>
<td>
-8.955
</td>
<td>
-7.361
</td>
<td>
-5.817
</td>
<td>
4000
</td>
<td>
397.501
</td>
<td>
634.025
</td>
<td>
1.014
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
0.812
</td>
<td>
1.707
</td>
<td>
-1.899
</td>
<td>
0.792
</td>
<td>
3.630
</td>
<td>
4000
</td>
<td>
464.329
</td>
<td>
811.650
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
-1.702
</td>
<td>
0.677
</td>
<td>
-2.747
</td>
<td>
-1.718
</td>
<td>
-0.560
</td>
<td>
4000
</td>
<td>
407.157
</td>
<td>
627.254
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
25.578
</td>
<td>
24.768
</td>
<td>
7.085
</td>
<td>
18.854
</td>
<td>
64.005
</td>
<td>
4000
</td>
<td>
724.652
</td>
<td>
1349.801
</td>
<td>
1.002
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
-0.017
</td>
<td>
0.045
</td>
<td>
-0.089
</td>
<td>
-0.017
</td>
<td>
0.058
</td>
<td>
4000
</td>
<td>
899.084
</td>
<td>
1886.383
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
-0.293
</td>
<td>
2.775
</td>
<td>
-5.001
</td>
<td>
-0.181
</td>
<td>
4.082
</td>
<td>
4000
</td>
<td>
479.522
</td>
<td>
945.263
</td>
<td>
1.012
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
2.266
</td>
<td>
2.389
</td>
<td>
-1.474
</td>
<td>
2.172
</td>
<td>
6.332
</td>
<td>
4000
</td>
<td>
435.935
</td>
<td>
856.752
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
-1.944
</td>
<td>
2.768
</td>
<td>
-6.764
</td>
<td>
-1.760
</td>
<td>
2.218
</td>
<td>
4000
</td>
<td>
363.131
</td>
<td>
676.664
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
-1.020
</td>
<td>
1.637
</td>
<td>
-3.883
</td>
<td>
-0.944
</td>
<td>
1.583
</td>
<td>
4000
</td>
<td>
373.263
</td>
<td>
618.815
</td>
<td>
1.011
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
0.327
</td>
<td>
1.720
</td>
<td>
-2.483
</td>
<td>
0.318
</td>
<td>
3.193
</td>
<td>
4000
</td>
<td>
475.384
</td>
<td>
719.197
</td>
<td>
1.005
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
-5.177
</td>
<td>
0.800
</td>
<td>
-6.473
</td>
<td>
-5.187
</td>
<td>
-3.846
</td>
<td>
4000
</td>
<td>
348.760
</td>
<td>
802.225
</td>
<td>
1.011
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
-0.909
</td>
<td>
0.771
</td>
<td>
-2.188
</td>
<td>
-0.880
</td>
<td>
0.313
</td>
<td>
4000
</td>
<td>
435.760
</td>
<td>
689.286
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
-0.388
</td>
<td>
1.297
</td>
<td>
-2.555
</td>
<td>
-0.359
</td>
<td>
1.655
</td>
<td>
4000
</td>
<td>
429.791
</td>
<td>
1055.949
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
kernel_04
</td>
<td>
-0.370
</td>
<td>
0.510
</td>
<td>
-1.206
</td>
<td>
-0.363
</td>
<td>
0.435
</td>
<td>
4000
</td>
<td>
430.828
</td>
<td>
697.776
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
11.014
</td>
<td>
10.008
</td>
<td>
2.825
</td>
<td>
8.163
</td>
<td>
28.308
</td>
<td>
4000
</td>
<td>
585.552
</td>
<td>
1519.491
</td>
<td>
1.006
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
0.048
</td>
<td>
0.047
</td>
<td>
-0.029
</td>
<td>
0.048
</td>
<td>
0.124
</td>
<td>
4000
</td>
<td>
788.307
</td>
<td>
904.095
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
0.010
</td>
<td>
0.043
</td>
<td>
-0.058
</td>
<td>
0.008
</td>
<td>
0.083
</td>
<td>
4000
</td>
<td>
875.405
</td>
<td>
1588.602
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
0.987
</td>
<td>
0.075
</td>
<td>
0.863
</td>
<td>
0.986
</td>
<td>
1.111
</td>
<td>
4000
</td>
<td>
903.248
</td>
<td>
1636.189
</td>
<td>
1.004
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
-0.020
</td>
<td>
0.040
</td>
<td>
-0.086
</td>
<td>
-0.021
</td>
<td>
0.047
</td>
<td>
4000
</td>
<td>
782.705
</td>
<td>
1194.150
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
1.091
</td>
<td>
0.068
</td>
<td>
0.980
</td>
<td>
1.090
</td>
<td>
1.202
</td>
<td>
4000
</td>
<td>
851.982
</td>
<td>
1528.908
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
