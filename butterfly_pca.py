import numpy as np
import pandas as pd
from numpy import linalg as LA

class Butterfly:
    def __init__(self, yields, tenors):
        self.yields = yields
        self.tenors = tenors

        self.y_dm = yields - yields.mean() # De-mean yields
        self.eig_v = self._eig_decomp()
        self.factors = self._pca_factors()
        self.residuals = self._factor_resid()

        self.dv01 = yields.apply(lambda yld: self._par_dv01(yld, yld.name)) # Find dv01 for each observation



    def _eig_decomp(self) -> pd.DataFrame:
        """
        Constructing covariance matrix of yields and finding the eigenvectors and eigenvalues.

        :return: first three eigenvectors
        """

        cov = self.y_dm.cov()                     # Construct covariance matrix
        eigenvalues, eigenvectors = LA.eigh(cov)  # Get eigenvalues, eigenvectors of covariance matrix
        eig_v_df = pd.DataFrame(eigenvectors, index=self.y_dm.columns)

        # Anchor direction of loadings to 30Y loading
        eig_v_df.iloc[:, -1] = eig_v_df.iloc[:, -1] * np.sign(eig_v_df.iloc[-1, -1])
        eig_v_df.iloc[:, -2] = eig_v_df.iloc[:, -2] * np.sign(eig_v_df.iloc[-1, -2])
        eig_v_df.iloc[:, -3] = eig_v_df.iloc[:, -3] * np.sign(eig_v_df.iloc[-1, -3])

        # LA.eigh returns eigenvalues in ascending order of largest - code below reverses order and relabels
        eig_v_df = eig_v_df[eig_v_df.columns[::-1]].rename(
            columns={list(eig_v_df)[-1]: 0, list(eig_v_df)[-2]: 1, list(eig_v_df)[-3]: 2}
        )

        return eig_v_df.iloc[:, [0, 1, 2]]

    def _pca_factors(self) -> tuple:
        """
        Constructing the factors for the PCA factor model (y_dm = alpha * e). eT * y_dm = alpha * e*eT = alpha * 1.

        :return: tuple of factor arrays
        """

        factor_1 = (self.eig_v.iloc[:, 0] * self.y_dm).sum(axis=1)
        factor_2 = (self.eig_v.iloc[:, 1] * self.y_dm).sum(axis=1)
        factor_3 = (self.eig_v.iloc[:, 2] * self.y_dm).sum(axis=1)
        return factor_1, factor_2, factor_3

    def _factor_resid(self) -> pd.Series:
        """
        Computing the 2nd factor residuals from PCA factor model for the most recent observation.

        :return: 2nd factor residuals array
        """

        current_y = self.y_dm.iloc[-1]
        current_factors = self.factors[0].iloc[-1], self.factors[1].iloc[-1], self.factors[2].iloc[-1]

        residuals = current_y - (current_factors[0]*self.eig_v.iloc[:, 0] + current_factors[1]*self.eig_v.iloc[:, 1])
        return residuals

    def construct_butterfly(self) -> tuple:
        """
        Creating the butterfly trade time series, hedged against changes in DV01 and changes in factor 1 and 2.
        This calculation follows from Huggins and Schaller 2nd ed. - ch. 3.

        :return: time series of butterfly trade
        """

        lower, belly, upper = self.tenors

        # Calculating PCA/DV01 hedge ratios
        A = np.array([[self.dv01[lower].iloc[-1]*self.eig_v.loc[lower, 0], self.dv01[upper].iloc[-1]*self.eig_v.loc[upper, 0]],
                      [self.dv01[lower].iloc[-1]*self.eig_v.loc[lower, 1], self.dv01[upper].iloc[-1]*self.eig_v.loc[upper, 1]]])

        B = np.array([[-self.dv01[belly].iloc[-1] * self.eig_v.loc[belly, 0]],
                      [-self.dv01[belly].iloc[-1] * self.eig_v.loc[belly, 1]]])

        hedge_ratios = LA.inv(A) @ B

        # Constructing butterfly trade performance (in bps), (equivalent to long butterfly hence -ve sign)
        # "how many bp of belly yield move would give the same P&L as this whole trade?"
        butterfly_series = ((-self.dv01[lower].iloc[-1] * hedge_ratios[0] * self.y_dm[lower]) +
                     (-self.dv01[belly].iloc[-1] * self.y_dm[belly]) +
                     (-self.dv01[upper].iloc[-1] * hedge_ratios[1] * self.y_dm[upper])
                     ) * (100/self.dv01[belly].iloc[-1])

        return butterfly_series, np.array([hedge_ratios[0][0], 1, hedge_ratios[1][0]])

    @staticmethod
    def _par_dv01(yld, tenor) -> pd.DataFrame:
        """
        Calculation from Truckman 4th ed.

        :return: par DV01
        """

        y = yld / 100
        T = float(tenor) # need to be careful with possible string inputs
        return (1/(100*y)) * (1 - 1/((1 + y/2)**(2*T)))


class Steepener(Butterfly):
    def __init__(self, yields, tenors):
        Butterfly.__init__(self, yields, tenors)
        assert(len(self.tenors) == 2)
        self.residuals = self._factor_resid()

    def _factor_resid(self) -> pd.Series:
        """
        Computing the 1st factor residuals from PCA factor model for the most recent observation.

        :return: 2nd factor residuals array
        """

        current_y = self.y_dm.iloc[-1]
        current_factors = self.factors[0].iloc[-1], self.factors[1].iloc[-1], self.factors[2].iloc[-1]

        residuals = current_y - (current_factors[0] * self.eig_v.iloc[:, 0])
        return residuals

    def construct_steepener(self) -> tuple:
        """
        Creating the steepener trade time series, hedged against changes in DV01 and changes in factor 1.
        This calculation follows from Huggins and Schaller 2nd ed. - ch. 3.

        :return: time series of steepener trade
        """

        lower, upper = self.tenors

        hedge_ratio = (self.dv01[upper].iloc[-1] / self.dv01[lower].iloc[-1]) * (self.eig_v.loc[upper, 0] / self.eig_v.loc[lower, 0])

        steepener = ((-self.dv01[lower].iloc[-1] * hedge_ratio * self.y_dm[lower]) +
                     (self.dv01[upper].iloc[-1] * 1 * self.y_dm[upper])
                     ) * (100/self.dv01[upper].iloc[-1])

        return steepener, np.array([hedge_ratio, -1])

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
    # butterfly = Butterfly(yields[[2, 3, 5, 7, 10, 30]], (3,10,30))
    # factors = butterfly.factors
    # butterfly_series, hedge_ratios = butterfly.construct_butterfly()

    steepener = Steepener(yields[[2, 3, 5, 7, 10, 30]], (3,10))
    factors = steepener.factors
    steepener_series, hedge_ratios = steepener.construct_steepener()

    print(hedge_ratios, steepener_series)


if __name__ == "__main__":
    main()