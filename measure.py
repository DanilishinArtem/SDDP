import torch


def Expectation(obj, grad, p):
    if p is not None:
        return (torch.matmul(p, obj), torch.matmul(p, grad))
    

def Expectation_AVaR(obj, grad, p, a, l, sense):
    n_samples, n_states = grad.shape
    if p is None:
        p = torch.ones(n_samples)/n_samples
    objAvg = torch.dot(p, obj)
    gradAvg = torch.dot(p, grad)
#    assert(type(gradAvg) == list and len(gradAvg) == len(p))
    objSortedIndex = torch.argsort(obj)
    if sense == -1:
        objSortedIndex = objSortedIndex[::-1]
    ## store the index of 1-alpha percentile ##
    tempSum = 0
    for index in objSortedIndex:
        tempSum += p[index]
        if tempSum >= 1 - a:
            kappa = index
            break
#    kappa = objSortedIndex[int((1 - a) * sampleSize)]
    ## obj=(1-lambda)*objAvg+lambda(obj_kappa+1/alpha*avg((obj_kappa - obj_l)+))
    objLP = (1 - l) * objAvg + l * obj[kappa]
    ## grad=(1-lambda)*gradAvg+lambda(grad_kappa+1/alpha*avg((pos))
    gradLP = (1 - l) * gradAvg + l * grad[kappa]

    gradTerm = torch.zeros((n_samples, n_states))
    objTerm = torch.zeros(n_samples)
    for j in range(n_samples):
        if sense*(obj[j] - obj[kappa]) >= 0:
            gradTerm[j] = sense * (grad[j] - grad[kappa])
            objTerm[j] = sense * (obj[j] - obj[kappa])
    objLP += sense * l * torch.dot(p, objTerm) / a
    gradLP += sense * l * torch.dot(p, gradTerm) / a
    return (objLP, gradLP)
