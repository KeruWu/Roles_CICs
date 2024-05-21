import torch
import torch.nn as nn

# Helper function for mean diff
def _diff_mean(X, Y, sample_weight=None):
    disp = nn.MSELoss()
    if sample_weight is None:
        return disp(torch.mean(X, axis=0), torch.mean(Y, axis=0))
    else:
        return disp(sample_weight.matmul(X) / torch.sum(sample_weight),
                    torch.mean(Y, axis=0))

# Helper function for Maximum mean discrepancy (MMD)
def _rbf_kernel(X, Y, sigma_list):
    m = X.size(0)

    Z = torch.cat((X, Y), 0)
    ZZT = torch.mm(Z, Z.t())
    diag_ZZT = torch.diag(ZZT).unsqueeze(1)
    Z_norm_sqr = diag_ZZT.expand_as(ZZT)
    # exponent[i,j] = ||Z[i] - Z[j]||_2^2
    exponent = Z_norm_sqr - 2 * ZZT + Z_norm_sqr.t()

    K = 0.0
    for sigma in sigma_list:
        gamma = 1.0 / (2 * sigma**2)
        K += torch.exp(-gamma * exponent)

    return K[:m, :m], K[:m, m:], K[m:, m:], len(sigma_list)



def _rbf_mmd2(X, Y, sample_weight=None, sigma_list = [0.1, 1, 10]):
    K_XX, K_XY, K_YY, d = _rbf_kernel(X, Y, sigma_list)

    m = K_XX.size(0)
    l = K_YY.size(0)

    if sample_weight is None:
        K_XX_sum = K_XX.sum()
        K_YY_sum = K_YY.sum()
        K_XY_sum = K_XY.sum()

        mmd2 = (K_XX_sum / (m * m)
              + K_YY_sum / (l * l)
              - 2.0 * K_XY_sum / (m * l))
    else:
        K_XX_sum = sample_weight.matmul(K_XX.matmul(sample_weight))
        K_YY_sum = K_YY.sum()
        K_XY_sum = (sample_weight.matmul(K_XY)).sum()

        s = sample_weight.sum()

        mmd2 = (K_XX_sum / (s * s)
              + K_YY_sum / (l * l)
              - 2.0 * K_XY_sum / (s * l))

    return mmd2

def freeze_bn_layers(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.eval()

def unfreeze_bn_layers(model):
     for m in model.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.train()