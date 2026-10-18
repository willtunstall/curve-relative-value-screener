import numpy as np
import pandas as pd
import statsmodels.api as sm

from butterfly_pca import Butterfly

class MeanRev:
    def __init__(self, butterfly_series):
        self.butterfly = butterfly_series

        self.butterfly_df = self._initialise_df()

        self.lamda, self.mu, self.sigma, self.half_life = self._ou_process()

        self.direction = 0
        self.target = 0

        self.future_paths_df = self.project_future_path(n=150)

    def _initialise_df(self) -> pd.DataFrame:
        """
        Initialise df.

        :return: data frame with historical trade data
        """
        # butterfly_df = pd.DataFrame(self.butterfly, columns=['butterfly_t'])
        butterfly_df = self.butterfly.rename('butterfly_t').to_frame()
        butterfly_df['butterfly_t+1'] = butterfly_df['butterfly_t'].shift(-1)
        butterfly_df.dropna(inplace=True)

        return butterfly_df

    def _ou_process(self) -> tuple:
        """
        Fit historical series to OU process and find parameters. **REVIEW TIME SERIES CHECKS**
        Calculations from Thijs van den Berg - Calibrating the Ornstein-Uhlenbeck (Vasicek) model (May 28, 2011)
        with a delta of 1.

        :return: OU parameters
        """

        s = self.butterfly_df['butterfly_t']
        s = sm.add_constant(s)
        s_1 = self.butterfly_df['butterfly_t+1']

        reg = sm.OLS(s_1, s) # AR(1)
        results = reg.fit()

        a = results.params.iloc[1]
        b = results.params.iloc[0]
        sd_error = np.sqrt(results.mse_resid)

        delta = 1 # time steps are consecutive days

        lamda = -(np.log(a) / delta)
        mu = b / (1 - a)
        sigma = sd_error * np.sqrt((-2*np.log(a))/(delta*(1-a**2)))
        half_life = np.log(2) / lamda

        return lamda, mu, sigma, half_life

    def _exp_future_path(self, s) -> tuple:
        """
        Finding the expected future path of series and the 2 sigma bands.

        :param s: current point in series
        :return: future point in series + 2 sigma bands
        """

        delta = 1

        s_i = s * np.exp(-self.lamda * delta) + self.mu*(1 - np.exp(-self.lamda * delta))

        sigma2 = self.sigma ** 2
        squared_noise = (sigma2 / (2 * self.lamda)) * (1 - np.exp(-2 * self.lamda * delta))
        upper_2 = s_i + 2*np.sqrt(squared_noise)
        lower_2 = s_i - 2*np.sqrt(squared_noise)

        return s_i, upper_2, lower_2

    def project_future_path(self, n) -> pd.DataFrame:
        """
        Project future paths of butterfly trade n-periods into the future.

        :param n: periods in future
        :return: dataframe with future paths t+1 -> t+n
        """

        future_paths_df = pd.DataFrame(np.zeros(shape=(n, 3), dtype=float),
                                       columns=['future_path', 'upper_2', 'lower_2'],
                                       index=pd.date_range(start=self.butterfly_df.index[-1],
                                                           periods=n,
                                                           freq='D')
                                       )

        idx = future_paths_df.index
        future_paths_df.loc[idx[0], 'future_path'] = self.butterfly.iloc[-1]

        for i in np.arange(future_paths_df.shape[0] - 1):
            t1 = i + 1
            # (future_paths_df['future_path'].iloc[t1], future_paths_df['upper_2'].iloc[t1],
            #  future_paths_df['lower_2'].iloc[t1]) = self._exp_future_path(s=future_paths_df['future_path'].iloc[i])

            (future_paths_df.loc[idx[t1], 'future_path'], future_paths_df.loc[idx[t1], 'upper_2'],
             future_paths_df.loc[idx[t1], 'lower_2']) = self._exp_future_path(s=future_paths_df.loc[idx[i], 'future_path'])

        return future_paths_df

    # def vectorised_project_future_path(self, n) -> pd.DataFrame:
    #     future_paths_df = pd.DataFrame(np.zeros(shape=(n, 3), dtype=float),
    #                                    columns=['future_path', 'upper_2', 'lower_2'],
    #                                    index=pd.date_range(start=self.butterfly_df.index[-1],
    #                                                        periods=n,
    #                                                        freq='D')
    #                                    )
    #
    #     future_paths_df['future_path'].iloc[0] = self.butterfly.iloc[-1]
    #
    #     delta = 1
    #
    #     s_i = s * np.exp(-self.lamda * delta) + self.mu * (1 - np.exp(-self.lamda * delta))
    #
    #     sigma2 = self.sigma ** 2
    #     squared_noise = (sigma2 / (2 * self.lamda)) * (1 - np.exp(-2 * self.lamda * delta))
    #     upper_2 = s_i + 2 * np.sqrt(squared_noise)
    #     lower_2 = s_i - 2 * np.sqrt(squared_noise)
    #
    #     future_paths_df['future_path'] =


    def vectorised_first_pass_sim(self, n) -> np.ndarray:
        """
        Vectorised Monte-carlo simulation of OU process to calculate hitting times. Target set to be value between
        mean and current observed value.

        :param n: paths into the future
        :return: hitting times
        """

        delta = 1
        sim_n = 10000  # number of simulations

        # Generate random noise
        z = np.random.normal(0, 1, size=(sim_n, n))

        s = (self.mu * (1 - np.exp(-self.lamda * delta)) +
             self.sigma * np.sqrt((1 / (2 * self.lamda)) * (1 - np.exp(-2 * self.lamda * delta))) * z)
        s[:, 0] += (np.exp(-self.lamda * delta) * self.butterfly.iloc[-1])

        for t in range(1, n):
            s[:,t] += (np.exp(-self.lamda * delta) * s[:, t-1])

        # setting target value as 1/2 diff between mean and observed
        self.target = self.butterfly.iloc[-1] + ((self.mu - self.butterfly.iloc[-1]) / 2)
        self.direction = np.sign(self.target - self.butterfly.iloc[-1])  # Direction of trade

        if self.direction == 1.0:
            crossed = s > self.target
        else:
            crossed = s < self.target

        hit = crossed.any(axis=1)  # paths that reach the target within n days
        first_cross = crossed.argmax(axis=1)  # index of the first True in each row
        hitting_times = first_cross[hit].astype(float)

        return hitting_times


    # def _simulated_path(self, s) -> float:
    #     """
    #     Simulate random path of OU process.
    #
    #     :param s: current point in series
    #     :return: simulated path at t+1
    #     """
    #
    #     delta = 1
    #     z_1 = np.random.normal(0, 1, size=1)
    #     s_1 = (np.exp(-self.lamda * (delta)) * s +
    #            self.mu * (1 - np.exp(-self.lamda * (delta))) +
    #            self.sigma * np.sqrt((1 / (2 * self.lamda)) * (1 - np.exp(-2 * self.lamda * (delta)))) * z_1)
    #
    #     return s_1
    #
    # def first_pass_sim(self, n) -> np.ndarray:
    #     """
    #     Monte-carlo simulation of OU process to calculate hitting times. Target set to be value between mean and
    #     current observed value.
    #
    #     :param n: periods in future
    #     :return: array of hitting times and direction of trade
    #     """
    #
    #     hitting_times = np.array([])
    #     sim_n = 10000  # number of simulations
    #
    #     # setting target value as 1/2 diff between mean and observed
    #     self.target = self.butterfly.iloc[-1] + ((self.mu - self.butterfly.iloc[-1]) / 2)
    #     self.direction = np.sign(self.target - self.butterfly.iloc[-1]) # Direction of trade
    #
    #     for _ in range(sim_n):
    #         path_sim = np.array([self.butterfly.iloc[-1]])
    #         for t in range(n):
    #             s_t_1 = self._simulated_path(s=path_sim[-1])  # simulate path t+1 from path t
    #             if self.direction == 1.0:
    #                 if s_t_1 > self.target:
    #                     hitting_times = np.append(hitting_times, t)  # record day of first passage
    #                     break
    #                 else:
    #                     path_sim = np.append(path_sim, s_t_1)
    #             else:
    #                 if s_t_1 < self.target:
    #                     hitting_times = np.append(hitting_times, t)
    #                     break
    #                 else:
    #                     path_sim = np.append(path_sim, s_t_1)
    #
    #     return hitting_times



def main():
    fed_data = pd.read_csv('feds200628.csv', skiprows=9)
    fed_data['Date'] = pd.to_datetime(fed_data.Date)
    fed_data.set_index('Date', inplace=True)

    par_yields_cols = [c for c in fed_data.columns if c.find('SVENPY') > -1]
    par_yields = fed_data[par_yields_cols]
    par_yields = par_yields.rename(columns=dict(zip(par_yields_cols, [int(c[-2:]) for c in par_yields_cols])))

    start_date = par_yields.iloc[-504].name
    end_date = par_yields.iloc[-1].name

    yields = par_yields[start_date:end_date].dropna()
    yields.columns.rename('Maturity', inplace=True)
    butterfly = Butterfly(yields[[2, 3, 5, 7, 10, 30]], tenors=(2, 5, 10))
    butterfly_series, hedge_ratios = butterfly.construct_butterfly()

    mean_rev = MeanRev(butterfly_series)
    future_paths_df = mean_rev.project_future_path(150)
    hitting_times = mean_rev.vectorised_first_pass_sim(n=150)

    # z = np.random.normal(0, 1, size=(10000, 150))
    print(hitting_times)





if __name__ == '__main__':
    main()