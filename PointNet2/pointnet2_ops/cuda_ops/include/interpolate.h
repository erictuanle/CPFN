#pragma once

#include <torch/extension.h>
#include <vector>

std::vector<at::Tensor> three_nn(at::Tensor unknowns, at::Tensor knows);
at::Tensor three_weighted_sum(at::Tensor points, at::Tensor idx,
                              at::Tensor weight);
at::Tensor three_weighted_sum_grad(at::Tensor grad_out, at::Tensor idx,
                                   at::Tensor weight, const int m);