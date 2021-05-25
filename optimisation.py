import torch
from torch import optim
import matplotlib.pyplot as plt

# choose the one you using. substitution in code needed
from IPython.display import clear_output
# from google.colab import output


def sigma(pred, old, f, p, q, log_dets, coef_ll = None, coef_var = 1):
    ans = 0
    if coef_ll is not None:
        ans += -coef_ll*(q.log_pdf(pred)+log_dets).sum()
    if coef_var is not None:
        ans += coef_var*((f(old)**2 *torch.exp(p.log_pdf(old)-q.log_pdf(pred)-log_dets)).mean())
    return ans

def estimate_params(p, q, f, model, sample_size = 10**3, lear_rt = 1e-3, 
                    epoch_amnt = 2*10**3, lr_downing_num=1, ll_coef = None, var_coef=1, dev = None,
                    todraw = True):
    lr = lear_rt #learning rate
    max_epochs = epoch_amnt
    samples_amnt = sample_size
    if dev is None:
        dev = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    xb = p.sampler(samples_amnt)
    xb = xb.to(dev)
    opt = optim.Adam(model.parameters(), lr=lr, weight_decay=1e-6)

    lambda2 = lambda itnum: 0.9
    scheduler = optim.lr_scheduler.MultiplicativeLR(opt, lr_lambda=lambda2)

    #best_model = None
    #wanted to deepcopy.....

    best_min_loss = float('Inf')

    for itnum in range(lr_downing_num):
        epoch_cnt = 0
        inc_res = 0
        cur_res = 10**6
        scheduler.step()
        is_nan_param = False

        # heuristic of stopping
        while True:
            if max_epochs is not None:
                if epoch_cnt > max_epochs:
                    break
            zb, log_dets = model(xb)
            log_dets = log_dets.squeeze()
            loss = sigma(zb, xb, f, p, q, log_dets)

            if torch.isnan(loss):
                print("nan loss")
                break

            loss.backward()
            opt.step()
            opt.zero_grad()

            is_nan_param = False
            for elem in model.parameters():
                if torch.any(torch.isnan(elem)).item():
                    print("nan param")
                    is_nan_param = True
                    break
            if is_nan_param:
                break

            if best_min_loss > loss.item():
                inc_res = 0
                best_min_loss = loss.item()

            if inc_res > 50:
                break

            prev_res = cur_res
            cur_res = loss
            epoch_cnt += 1
            inc_res += 1

            if todraw and epoch_cnt % 1000 == 0:
                clear_output(wait=True)
                # output.clear()
                print("Cur iter:", itnum)
                print("Curr_loss:", loss)
                print("Min_loss:", best_min_loss)
                print("Current epochs:", epoch_cnt)
                print("Current tolerance:", torch.abs(cur_res-prev_res))
                plotting = zb.detach().cpu()
                color = (q.log_pdf(zb)+log_dets).detach().cpu()
                plt.scatter(plotting[:, 0], plotting[:, 1], c=color )
                cbar = plt.colorbar()
                plt.grid()
                plt.show()
        if is_nan_param:
            break


