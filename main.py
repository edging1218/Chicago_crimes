from data import Data
# import vis
from model import Model
from time import time


if __name__ == '__main__':
    crimes = Data()
    # vis.plot_map_contour(crimes.data, 'Month')
    # vis.plot_bar(crimes.data, 'Hour')
    # vis.plot_heatmap(crimes.data, 'Location Description')
    # vis.plot_bar(crimes.data, 'Month')
    # vis.plot_heatmap(crimes.data, 'Month')
    # vis.biplot(crimes.data, 'Month')
    # vis.biplot(crimes.data, 'Hour')
    # vis.biplot(crimes.data, 'Community Area', crimes.community_name)
    # vis.plot_time(crimes.data)
    crimes.random_sample(0.2)
    crimes.preprocessing()

    # param_logit_grid = {'logit_grid':
    #                         {'penalty': ['l1', 'l2'],
    #                          'C': [10 ** i for i in range(-6, 1, 2)]}}
    # logit = Model(crimes, 'logit', param_logit_grid)
    # logit.grid_search_all('accuracy', 4)


    # param_logit = {'logit': {'penalty': 'l2'}}
    # logit = Model(crimes, 'logit', param_logit)
    # start = time()
    # logit.run_all()
    # end = time()
    # print 'Time used: {}.'.format((end-start)/60)

    start = time()
    param_xgb = {'xgb_grid': {'learning_rate': [0.01, 0.05, 0.1],
                         'n_estimators': [200, 300, 400]}}
    xgb = Model(crimes, 'xgb', param_xgb)
    res = xgb.grid_search_all('neg_log_loss', 3)
    xgb.run_all()
    end = time()
    print 'Time used: {}.'.format((end-start)/60)


