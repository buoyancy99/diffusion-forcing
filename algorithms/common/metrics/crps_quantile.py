import numpy as np
import torch
from gluonts.nursery.few_shot_prediction.src.meta.metrics.crps import CRPS


quantiles = (np.arange(20)/20.0)[1:]
crps_quantile = CRPS(quantiles=quantiles)


def crps_quantile_sum(pred, truth):
    """
    Sum along the feature dimension of the time series, then compute CRPS and take average.
    :param pred: (samples, time, batch, feature)
    :param truth: (time, batch, feature)
    :return: torch.Tensor holding a float
    """

    crps_quantile.reset()
    pred = pred.sum(dim=-1)
    truth = truth.sum(dim=-1)

    # For each quantile, calculate the value across the sample dimension
    pred_quantiles = torch.stack([pred.kthvalue(int(q * pred.shape[0]), dim=0)[0] for q in quantiles], dim=2)

    pred_quantiles = pred_quantiles.permute((1, 0, 2))  # (batch, time, quantiles)
    truth = truth.transpose(0, 1)  # (batch, time)
    mask = torch.ones_like(truth)

    crps_quantile.update(pred_quantiles, truth, mask)
    return crps_quantile.compute()


if __name__ == '__main__':
    samples = 100
    time = 50
    batch = 10
    feature = 23

    y_pred = torch.rand(samples, time, batch, feature)
    y_true = torch.rand(time, batch, feature)
    crps_sum = crps_quantile_sum(y_pred, y_true)

    # GluonTS m_sum_mean_wQuantileLoss:
    from gluonts.model.forecast import SampleForecast
    from gluonts.evaluation import MultivariateEvaluator
    import pandas as pd


    y_pred = y_pred.numpy()
    y_true = y_true.numpy()

    forecasts = []
    tss = []
    period_index = pd.period_range(start="2020-01-01", periods=y_true.shape[0], freq="1H")

    # Prepare data for MultivariateEvaluator
    for i in range(batch):
        forecast = SampleForecast(
            samples=y_pred[:, :, i, :],
            start_date=period_index[0],
            item_id=str(i),
        )
        forecasts.append(forecast)
        ts_df = pd.DataFrame(y_true[:, i, :], index=period_index)
        tss.append(ts_df)

    evaluator = MultivariateEvaluator(quantiles=quantiles, target_agg_funcs={'sum': np.sum})
    agg_metrics, item_metrics = evaluator(iter(tss), iter(forecasts))

    print("- " * 40)
    print("CRPS from custom function:", crps_sum.item())
    print("wQuantileLoss from GluonTS:", agg_metrics['m_sum_mean_wQuantileLoss'])
    print(crps_sum.item() - agg_metrics['m_sum_mean_wQuantileLoss'])
    print(abs(crps_sum.item() - agg_metrics['m_sum_mean_wQuantileLoss']) < 1e-3)  # We only report 3 decimals
