import os

from dash import Dash, html, dcc
from dash.dependencies import Input, Output, State
import dash_bootstrap_components as dbc

import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

import numpy as np
import pandas as pd
import locale


current_locale = locale.setlocale( locale.LC_ALL, 'en_CA.UTF-8' )


class TimeValueOfMoney:
    # assume 'times' and 'cash_flows' columns
    def __init__(
        self,
        interest_rate,
        time_cash_flows_df,
    ):
        self.ACCURACY = 0.00005
        self.MAX_ITERATIONS = 200
        self.interest_rate = interest_rate
        # print(self.interest_rate)
        self.time_cash_flows_df = time_cash_flows_df
        # print(self.time_cash_flows_df)

    def loan_pv(self, interest_rate=None):
        if interest_rate is None:
            interest_rate = self.interest_rate
        pv = 0
        for row in self.time_cash_flows_df.itertuples():
            pv += row.cash_flows / ((1 + interest_rate) ** row.times)
        return pv

    def loan_ytm(self):
        bottom = 0
        top = 1

        pv = self.loan_pv()

        while (self.loan_pv(top) > pv):
            top = top * 2
        ytm = 0.5 * (top + bottom)
        for i in range(self.MAX_ITERATIONS):
            diff = self.loan_pv(ytm) - pv
            if (abs(diff) < self.ACCURACY):
                return ytm
            if (diff > 0):
                bottom = ytm
            else:
                top = ytm
            ytm = 0.5 * (top + bottom)

        return ytm

    def loan_ytm_from_price(self, pv):
        bottom = 0
        top = 1

        while (self.loan_pv(top) > pv):
            top = top * 2
        ytm = 0.5 * (top + bottom)
        for i in range(self.MAX_ITERATIONS):
            diff = self.loan_pv(ytm) - pv
            if (abs(diff) < self.ACCURACY):
                return ytm
            if (diff > 0):
                bottom = ytm
            else:
                top = ytm
            ytm = 0.5 * (top + bottom)

        return ytm

    def loan_duration(self, interest_rate):
        pv = self.loan_pv()

        duration_sum = 0
        for row in self.time_cash_flows_df.itertuples():
            duration_sum += row.times * row.cash_flows / ((1 + interest_rate) ** row.times)

        return duration_sum / pv

    def loan_mac_duration(self):
        ytm = self.loan_ytm()
        duration = self.loan_duration(ytm)

        return duration

    def loan_mac_duration_from_price(self, pv):
        ytm = self.loan_ytm_from_price(pv)
        duration = self.loan_duration(ytm)

        return duration

    def loan_mod_duration(self):
        ytm = self.loan_ytm()
        duration = self.loan_duration(ytm)

        return duration / (1 + ytm)

    def loan_mod_duration_from_price(self, pv):
        ytm = self.loan_ytm_from_price(pv)
        duration = self.loan_duration(ytm)

        return duration / (1 + ytm)

    def loan_convexity(self):
        pv = self.loan_pv()

        convex_sum = 0
        for row in self.time_cash_flows_df.itertuples():
            convex_sum += row.cash_flows * row.times * (row.times + 1) / ((1 + self.interest_rate) ** row.times)

        return convex_sum / ((1 + self.interest_rate) ** 2) / pv

    def loan_pv_delta_on_interest_rate_delta(self, interest_rate_delta):
        pv = self.loan_pv()
        duration = self.loan_mod_duration()
        delta_percent = -duration * interest_rate_delta

        return delta_percent * pv

    def loan_pv_delta_on_interest_rate_delta_with_convex(self, interest_rate_delta):
        pv = self.loan_pv()
        # print(pv)
        duration = self.loan_mod_duration()
        # print(duration)
        convexity = self.loan_convexity()
        # print(convexity)
        delta_percent = -duration * interest_rate_delta
        # print(delta_percent)
        delta_percent += (convexity / 2) * (interest_rate_delta ** 2)
        # print(delta_percent)

        return delta_percent * pv


class SimpleLoanCalculator:
    def __init__(
        self,
        loan_principal,
        loan_interest_rate,
        market_interest_rate,
        origination_fee,
        origination_fee_separate,
        initial_cvl,
        margin_call_cvl,
        liquidation_cvl,
        btc_price,
        start_date,
        duration,
        day_count_convention,
    ):
        ################### Initial Inputs ###################
        if day_count_convention.startswith('30/'):
            due_dates = pd.date_range(start_date, periods=duration+1, freq='30D').tolist()[1:]
        elif day_count_convention.startswith('actual/'):
            due_dates = pd.date_range(start_date, periods=duration, freq='M').tolist()
        else:
            raise RuntimeError(f"Unsupported basis: '{day_count_convention}'")

        end_date = due_dates[-1]
        period_in_days = (end_date - start_date).days / len(due_dates)

        if day_count_convention.endswith('/360'):
            days_per_year = 360
        elif day_count_convention.endswith('/365'):
            days_per_year = 365
        elif day_count_convention.endswith('/actual'):
            days_per_year = (end_date - start_date).days
        else:
            raise RuntimeError(f"Unsupported basis: '{day_count_convention}'")

        interest_schedule = [start_date] + due_dates
        interest_schedule_df = pd.DataFrame(interest_schedule, columns=['PeriodDueDate'])
        interest_schedule_df['PeriodStartDate'] = interest_schedule_df['PeriodDueDate'].shift(1)
        interest_schedule_df['DaysInPeriod'] = (interest_schedule_df['PeriodDueDate'] - interest_schedule_df['PeriodStartDate']).dt.days
        # interest_schedule_df.drop(index=0, inplace=True)
        interest_schedule_df.drop(columns=['PeriodStartDate'], inplace=True)

        ################### Outputs ###################
        dcf = self.simplified_day_count_factor(start_date, end_date, day_count_convention)
        dsirf = self.daily_simple_interest_rate_factor(day_count_convention)
        origination_fee_value = origination_fee * loan_principal
        daily_simple_interest_rate = loan_interest_rate * dsirf
        daily_interest = loan_principal * daily_simple_interest_rate
        monthly_30_day_interest = 30 * daily_interest
        if origination_fee_separate:
            # apr = [origination_fee + loan_interest_rate * duration * dcf]
            apr = [origination_fee + loan_interest_rate * dcf]
        else:
            # min_apr = origination_fee * loan_interest_rate * duration * dcf
            # max_apr =             1.0 * loan_interest_rate * duration * dcf
            min_apr = origination_fee * loan_interest_rate * dcf
            max_apr =             1.0 * loan_interest_rate * dcf
            apr = [min_apr, max_apr]
        # no fees
        self.apr_no_origination_fee = loan_interest_rate * dcf
        # annualized
        annualized_apr = [a**dcf for a in apr]
        final_payment = loan_principal + monthly_30_day_interest
        # use actual apr
        finance_charge = [a*loan_principal for a in apr]

        collateral = loan_principal / btc_price * initial_cvl
        collateral_value = loan_principal * initial_cvl
        first_margin_call_price = btc_price / initial_cvl * margin_call_cvl
        liquidation_price = btc_price / initial_cvl * liquidation_cvl

        # interest schedule
        interest_schedule_df['InterestDue'] = loan_principal * daily_simple_interest_rate * interest_schedule_df['DaysInPeriod']
        interest_schedule_df['Disbursement'] = np.float64(0)
        interest_schedule_df['Payment'] = np.float64(0)

        # start date disbursement & end date bullet payment
        interest_schedule_df.loc[interest_schedule_df['PeriodDueDate'] == start_date, 'Disbursement'] = -loan_principal
        if origination_fee_separate:
            interest_schedule_df.loc[interest_schedule_df['PeriodDueDate'] == start_date, 'Payment'] = origination_fee_value
        interest_schedule_df.loc[interest_schedule_df['PeriodDueDate'] == end_date, 'Payment'] = loan_principal

        interest_schedule_df.fillna(0, inplace=True)
        # interest_schedule_df.set_index('PeriodDueDate', inplace=True)
        self.interest_schedule_df = interest_schedule_df

        interest_schedule_df['duration_in_days'] = (interest_schedule_df['PeriodDueDate'] - start_date).dt.days
        interest_schedule_df['duration_in_years'] = interest_schedule_df['duration_in_days'] / 365.0
        interest_schedule_df['cashflow'] = interest_schedule_df['InterestDue'] + interest_schedule_df['Payment']

        tvm = TimeValueOfMoney(
            market_interest_rate / 365.0,
            interest_schedule_df.loc[:, ['duration_in_days','cashflow',]].rename(columns={'duration_in_days': 'times','cashflow': 'cash_flows',})
        )

        interest_schedule_df['present_value'] = interest_schedule_df['cashflow'] / (1 + market_interest_rate) ** interest_schedule_df['duration_in_years']
        interest_schedule_df['pv_sum'] = interest_schedule_df['present_value'].sum()
        interest_schedule_df['npv'] = interest_schedule_df['present_value'].sum() - loan_principal
        interest_schedule_df['present_value_percent'] = interest_schedule_df['present_value'] / interest_schedule_df['present_value'].sum()
        interest_schedule_df['pv_percent_sum'] = interest_schedule_df['present_value_percent'].sum()
        interest_schedule_df['duration_x_pv_percent'] = interest_schedule_df['duration_in_years'] * interest_schedule_df['present_value_percent']
        interest_schedule_df['mac_d'] = interest_schedule_df['duration_x_pv_percent'].sum()

        pv = tvm.loan_pv()
        npv = pv - loan_principal
        ytm = tvm.loan_ytm() * 365
        mac_d = tvm.loan_mac_duration() / 365.0
        mod_d = tvm.loan_mod_duration() / 365.0
        ytm_from_price = tvm.loan_ytm_from_price(loan_principal) * 365
        mac_d_from_price = tvm.loan_mac_duration_from_price(loan_principal) / 365.0
        mod_d_from_price = tvm.loan_mod_duration_from_price(loan_principal) / 365.0
        conv = tvm.loan_convexity() / 365.0 / 365.0
        market_interest_rate_delta = 0.01
        dv01 = tvm.loan_pv_delta_on_interest_rate_delta_with_convex(market_interest_rate_delta / 365.0)
        tvm.interest_rate += market_interest_rate_delta / 365.0
        pv_prime = tvm.loan_pv()
        dv_m_01 = tvm.loan_pv_delta_on_interest_rate_delta_with_convex(-market_interest_rate_delta / 365.0)
        tvm.interest_rate -= (2 * market_interest_rate_delta) / 365.0
        pv_m_prime = tvm.loan_pv()

        self.data = [
            ['-----= Inputs =-----', '-----= Inputs =-----'],
            ['Loan Principal', locale.currency(loan_principal, grouping=True)],
            ['Loan Interest Rate', '{:,.2f}%'.format(loan_interest_rate * 100)],
            ['Market Interest Rate', '{:,.2f}%'.format(market_interest_rate * 100)],
            ['Origination fee', '{:,.2f}%'.format(origination_fee * 100)],
            ['Origination Separate?', ('No', 'Yes')[origination_fee_separate]],
            ['Collateral-to-loan-value [CVL]', '{:,.2f}%'.format(initial_cvl * 100)],
            ['Margin Call CVL', '{:,.2f}%'.format(margin_call_cvl * 100)],
            ['Liquidation CVL', '{:,.2f}%'.format(liquidation_cvl * 100)],
            ['Day Count Convention', day_count_convention],
            ['Period (avg # of days)', '{:,.2f} days'.format(period_in_days)],
            ['Duration (# of months)', duration],
            ['Year duration (# of days)', days_per_year],
            ['Day Count factor', '{:,.4f}'.format(dcf)],
            ['-----= Outputs =-----', '-----= Outputs =-----'],
            ['BTC Price', locale.currency(btc_price, grouping=True)],
            ['Collateral', '{:,.2f} BTC'.format(collateral)],
            ['Collateral Value', locale.currency(collateral_value, grouping=True)],
            ['Annual Percentage Rate Min', '{:,.2f}%'.format(annualized_apr[0] * 100)],
            ['Annual Percentage Rate Max', '{:,.2f}%'.format(annualized_apr[-1] * 100)],
            ['Origination fee value', locale.currency(origination_fee_value, grouping=True)],
            ['Daily Interest Rate', '{:,.4f}%'.format(daily_simple_interest_rate * 100)],
            ['Daily Interest', locale.currency(daily_interest, grouping=True)],
            ['Monthly (30d) Interests', locale.currency(monthly_30_day_interest, grouping=True)],
            ['Final payment', locale.currency(final_payment, grouping=True)],
            ['Finance charge Min', locale.currency(finance_charge[0], grouping=True)],
            ['Finance charge Max', locale.currency(finance_charge[-1], grouping=True)],
            ['First margin call price', locale.currency(first_margin_call_price, grouping=True)],
            ['Liquidation price', locale.currency(liquidation_price, grouping=True)],
            ['Present Value', locale.currency(pv, grouping=True)],
            ['Net Present Value', locale.currency(npv, grouping=True)],
            ['Yield to Maturity', '{:,.4f}%'.format(ytm * 100)],
            ['Macaulay Duration', '{:,.2f} years'.format(mac_d)],
            ['Modified Duration', '{:,.4f}%'.format(mod_d)],
            ['Convexity', '{:,.4f}'.format(conv)],
            ['Value change on +{:,.2f}% i.r. change'.format(market_interest_rate_delta * 100), locale.currency(dv01, grouping=True)],
            ['Present Value @ {:,.2f}%'.format((market_interest_rate + market_interest_rate_delta) * 100), locale.currency(pv_prime, grouping=True)],
            ['Value change on -{:,.2f}% i.r. change'.format(market_interest_rate_delta * 100), locale.currency(dv_m_01, grouping=True)],
            ['Present Value @ {:,.2f}%'.format((market_interest_rate - market_interest_rate_delta) * 100), locale.currency(pv_m_prime, grouping=True)],
            ['Yield to Maturity @ par', '{:,.4f}%'.format(ytm_from_price * 100)],
            ['Macaulay Duration @ par', '{:,.2f} years'.format(mac_d_from_price)],
            ['Modified Duration @ par', '{:,.4f}%'.format(mod_d_from_price)],
        ]

    def table_df(self):
        return pd.DataFrame(self.data, columns=['Field', 'Value'])

    def schedule_df(self):
        return self.interest_schedule_df

    def simplified_day_count_factor(self, start_date, end_date, convention):
        # 30/360         - calculates the daily interest using a 360-day year and then multiplies that by 30 (standardized month).
        # 30/365         - calculates the daily interest using a 365-day year and then multiplies that by 30 (standardized month).
        # actual/360     - calculates the daily interest using a 360-day year and then multiplies that by the actual number of days in each time period.
        # actual/365     - calculates the daily interest using a 365-day year and then multiplies that by the actual number of days in each time period.
        # actual/actual  - calculates the daily interest using the actual number of days in the year and then multiplies that by the actual number of days in each time period.
        result = 0
        if convention == '30/360':
            # result = 30.0/360.0
            result = (end_date - start_date).days/360.0
        elif convention == 'actual/360':
            result = (end_date - start_date).days/360.0
        elif convention == '30/365':
            # result = 30.0/365.0
            result = (end_date - start_date).days/365.0
        elif convention == 'actual/365':
            result = (end_date - start_date).days/365.0
        elif convention == 'actual/actual':
            result = 1.0
        else:
            raise RuntimeError(f"Unsupported convention: '{convention}'")
        return result

    def daily_simple_interest_rate_factor(self, convention):
        result = 0
        if convention == '30/360' or convention == 'actual/360':
            result = 1.0/360.0
        elif convention == '30/365' or convention == 'actual/365':
            result = 1.0/365.0
        elif convention == 'actual/actual':
            result = 1.0/365.25
        else:
            raise RuntimeError(f"Unsupported convention: '{convention}'")
        return result

    def real_day_count_factor(self, start_date, end_date, convention):
        '''
            https://en.wikipedia.org/wiki/Day_count_convention
            https://www.investopedia.com/terms/d/daycount.asp
            https://www.investopedia.com/ask/answers/06/daycountconvention.asp
        '''
        # print(f"DCF for ({start_date}, {end_date}, {convention})")

        result = 0
        if convention == 'actual/360':
            result = (end_date - start_date).days/360.0
        elif convention == 'actual/365':
            result = (end_date - start_date).days/365.0
        elif convention == 'actual/actual':
            if start_date.year != end_date.year:
                start_of_to_year = pd.Timestamp(end_date.year, 1, 1)
                end_of_start_year = pd.Timestamp(start_date.year, 12, 31)
                result = (end_of_start_year - start_date).days/ \
                (365.0, 366.0)[start_date.is_leap_year] \
                + (int(end_date.year) - int(start_date.year) - 1) + \
                (end_date - start_of_to_year).days/ \
                (365.0, 366.0)[end_date.is_leap_year]
            else:
                result = (end_date - start_date).days/ \
                (365.0, 366.0)[end_date.is_leap_year]
        elif convention == '30/360' or convention == '30/365':
            day_per_year = 360.0 if convention == '30/360' else 365.0
            d1, d2 = start_date.day, end_date.day
            if d1 == 31:
                d1 -= 1
            if d2 == 31:
                d2 -= 1
            result = (int(d2) - int(d1)) + \
            30.0*(int(end_date.month) - int(start_date.month)) + \
            day_per_year*(int(end_date.year) - int(start_date.year))
            # print(f"#days used for '30/{day_per_year}': {result}")
            result = result / day_per_year
        else:
            raise RuntimeError(f"Unsupported convention: '{convention}'")
        return result

class CvlCalculator:
    def __init__(
        self,
        apr,
        mc_to_liq_time_period,
        mc_to_liq_confidence_level,
        mc_to_liq_time_horizon,
        mc_to_liq_var_not_cvar,
        init_to_mc_time_period,
        init_to_mc_confidence_level,
        init_to_mc_time_horizon,
        init_to_mc_var_not_cvar
    ):
        #
        # Raw Data & time resampling
        #
        my_dir = os.path.dirname(__file__)
        file_path = os.path.join(my_dir, 'BTC-USD_20140917-20250210.csv')
        # print(file_path)

        try:
            btcusd_1min = pd.read_csv(file_path, sep=",", header=0, names=["date","open","high","low","close","adj_close","volume"], index_col="date", parse_dates=True)
            print(btcusd_1min)

            # daily close & forward fill
            btcusd_1day_px = btcusd_1min.resample('1D')['close'].last().ffill().bfill()
            # 3 days close & forward fill
            btcusd_3day_px = btcusd_1min.resample('3D')['close'].last().ffill().bfill()
            # weekly close & forward fill
            btcusd_1week_px = btcusd_1min.resample('1W')['close'].last().ffill().bfill()
            # 2-weekly close & forward fill
            btcusd_2week_px = btcusd_1min.resample('2W')['close'].last().ffill().bfill()
            # 3-weekly close & forward fill
            btcusd_3week_px = btcusd_1min.resample('3W')['close'].last().ffill().bfill()
            # monthly close & forward fill
            btcusd_1month_px = btcusd_1min.resample('1M')['close'].last().ffill().bfill()
            # # 3-monthly close & forward fill
            # btcusd_3month_px = btcusd_1min.resample('3M')['close'].last().ffill().bfill()
            # # 6-monthly close & forward fill
            # btcusd_6month_px = btcusd_1min.resample('6M')['close'].last().ffill().bfill()
            # # yearly close & forward fill
            # btcusd_1year_px = btcusd_1min.resample('1YE')['close'].last().ffill().bfill()
            message = "success!"

            #
            # Log-returns
            #
            btcusd_1day_log_ret = np.log(btcusd_1day_px / btcusd_1day_px.shift(1)).dropna()
            btcusd_3day_log_ret = np.log(btcusd_3day_px / btcusd_3day_px.shift(1)).dropna()
            btcusd_1week_log_ret = np.log(btcusd_1week_px / btcusd_1week_px.shift(1)).dropna()
            btcusd_2week_log_ret = np.log(btcusd_2week_px / btcusd_2week_px.shift(1)).dropna()
            btcusd_3week_log_ret = np.log(btcusd_3week_px / btcusd_3week_px.shift(1)).dropna()
            btcusd_1month_log_ret = np.log(btcusd_1month_px / btcusd_1month_px.shift(1)).dropna()
            # btcusd_3month_log_ret = np.log(btcusd_3month_px / btcusd_3month_px.shift(1)).dropna()
            # btcusd_6month_log_ret = np.log(btcusd_6month_px / btcusd_6month_px.shift(1)).dropna()
            # btcusd_1year_log_ret = np.log(btcusd_1year_px / btcusd_1year_px.shift(1)).dropna()

            ################### Initial Inputs ###################
            if mc_to_liq_time_period.startswith('1-day'):
                mc_to_liq_btcusd_log_ret = btcusd_1day_log_ret
            elif mc_to_liq_time_period.startswith('3-day'):
                mc_to_liq_btcusd_log_ret = btcusd_3day_log_ret
            elif mc_to_liq_time_period.startswith('1-week'):
                mc_to_liq_btcusd_log_ret = btcusd_1week_log_ret
            else:
                raise RuntimeError(f"Unsupported basis: '{mc_to_liq_time_period}'")

            if mc_to_liq_confidence_level.startswith('99.99%'):
                mc_to_liq_cl = 100 - 99.99
            elif mc_to_liq_confidence_level.startswith('99.9%'):
                mc_to_liq_cl = 100 - 99.9
            elif mc_to_liq_confidence_level.startswith('99%'):
                mc_to_liq_cl = 100 - 99
            elif mc_to_liq_confidence_level.startswith('95%'):
                mc_to_liq_cl = 100 - 95
            else:
                raise RuntimeError(f"Unsupported basis: '{mc_to_liq_confidence_level}'")

            if init_to_mc_time_period.startswith('1-week'):
                init_to_mc_btcusd_log_ret = btcusd_1week_log_ret
            elif init_to_mc_time_period.startswith('2-weeks'):
                init_to_mc_btcusd_log_ret = btcusd_2week_log_ret
            elif init_to_mc_time_period.startswith('3-weeks'):
                init_to_mc_btcusd_log_ret = btcusd_3week_log_ret
            elif init_to_mc_time_period.startswith('1-month'):
                init_to_mc_btcusd_log_ret = btcusd_1month_log_ret
            else:
                raise RuntimeError(f"Unsupported basis: '{init_to_mc_time_period}'")

            if init_to_mc_confidence_level.startswith('99.99%'):
                init_to_mc_cl = 100 - 99.99
            elif init_to_mc_confidence_level.startswith('99.9%'):
                init_to_mc_cl = 100 - 99.9
            elif init_to_mc_confidence_level.startswith('99%'):
                init_to_mc_cl = 100 - 99
            elif init_to_mc_confidence_level.startswith('95%'):
                init_to_mc_cl = 100 - 95
            else:
                raise RuntimeError(f"Unsupported basis: '{init_to_mc_confidence_level}'")

            #
            # Volatility
            #
            mc_to_liq_vol = mc_to_liq_btcusd_log_ret.tail(mc_to_liq_time_horizon).std()
            init_to_mc_vol = init_to_mc_btcusd_log_ret.tail(init_to_mc_time_horizon).std()

            #
            # VaR
            #
            mc_to_liq_var = np.percentile(mc_to_liq_btcusd_log_ret.tail(mc_to_liq_time_horizon), mc_to_liq_cl)
            init_to_mc_var = np.percentile(init_to_mc_btcusd_log_ret.tail(init_to_mc_time_horizon), init_to_mc_cl)

            #
            # CVaR
            #
            mc_to_liq_cvar = mc_to_liq_btcusd_log_ret[mc_to_liq_btcusd_log_ret < mc_to_liq_var].mean()
            init_to_mc_cvar = init_to_mc_btcusd_log_ret[init_to_mc_btcusd_log_ret < init_to_mc_var].mean()

            #
            # Expected Loss
            #
            if mc_to_liq_var_not_cvar:
                mc_to_liq_loss = abs(mc_to_liq_var)
            else:
                mc_to_liq_loss = abs(mc_to_liq_cvar)

            if init_to_mc_var_not_cvar:
                init_to_mc_loss = abs(init_to_mc_var)
            else:
                init_to_mc_loss = abs(init_to_mc_cvar)

            #
            # Liquidation CVL / LTV
            #
            liquidation_cvl = 1 + apr
            liquidation_ltv = 1 / liquidation_cvl

            #
            # Margin Call CVL / LTV
            #
            margin_call_cvl = 1 + apr + mc_to_liq_loss
            margin_call_ltv = 1 / margin_call_cvl

            #
            # Initial CVL / LTV
            #
            initial_cvl = 1 + apr + mc_to_liq_loss + init_to_mc_loss
            initial_ltv = 1 / initial_cvl

            self.data = [
                ['', ''],
                ['-----= CVLs / LTVs Calculations =-----', ''],
                ['-----= Inputs =-----', ''],
                ['Liquidation Recovery APR', '{:,.2f}%'.format(apr * 100)],
                ['', ''],
                ['Margin Call - Time Period', mc_to_liq_time_period],
                ['Margin Call - Confidence Level', mc_to_liq_confidence_level],
                ['Margin Call - Time Horizon', '{:,.0f} samples'.format(mc_to_liq_time_horizon)],
                ['Margin Call - Loss estimation using: ', ('CVaR', 'Var')[mc_to_liq_var_not_cvar]],
                ['Margin Call - Implied Volatility', '{:,.2f}%'.format(mc_to_liq_vol * 100)],
                ['', ''],
                ['Initial - Time Period', init_to_mc_time_period],
                ['Initial - Confidence Level', init_to_mc_confidence_level],
                ['Initial - Time Horizon', '{:,.0f} samples'.format(init_to_mc_time_horizon)],
                ['Initial - Loss estimation using: ', ('CVaR', 'Var')[init_to_mc_var_not_cvar]],
                ['Initial - Implied Volatility', '{:,.2f}%'.format(init_to_mc_vol * 100)],

                ['-----= Outputs =-----', ''],
                ['-----= CVLs =-----', ''],
                ['Initial CVL', '{:,.0f}%'.format(initial_cvl * 100)],
                ['Margin Call CVL', '{:,.0f}%'.format(margin_call_cvl * 100)],
                ['Liquidation CVL', '{:,.0f}%'.format(liquidation_cvl * 100)],

                ['-----= LTVs =-----', ''],
                ['Initial LTV', '{:,.0f}%'.format(initial_ltv * 100)],
                ['Margin Call LTV', '{:,.0f}%'.format(margin_call_ltv * 100)],
                ['Liquidation LTV', '{:,.0f}%'.format(liquidation_ltv * 100)],

                ['-----= Details =-----', ''],
                ['Margin Call - Vol', '{:,.2f}%'.format(mc_to_liq_vol * 100)],
                ['Margin Call - VaR', '{:,.2f}%'.format(mc_to_liq_var * 100)],
                ['Margin Call - CVaR', '{:,.2f}%'.format(mc_to_liq_cvar * 100)],
                ['Margin Call - Expected Loss', '{:,.2f}%'.format(mc_to_liq_loss * 100)],
                ['', ''],
                ['Initial - Vol', '{:,.2f}%'.format(init_to_mc_vol * 100)],
                ['Initial - VaR', '{:,.2f}%'.format(init_to_mc_var * 100)],
                ['Initial - CVaR', '{:,.2f}%'.format(init_to_mc_cvar * 100)],
                ['Initial - Expected Loss', '{:,.2f}%'.format(init_to_mc_loss * 100)],
                ['', ''],
            ]

        except Exception as e:
            message = f"Exception: {e}"
            self.data = [
                ['-----= CVLs / LTVs Calculations =-----', ''],
                ['file_path', file_path],
                ['message', message],
            ]

            return

    def table_df(self):
        return pd.DataFrame(self.data, columns=['Field', 'Value'])


def calc_table_height(df, base=50, height_per_row=20, char_limit=30, height_padding=19):
    '''
    df: The dataframe with only the columns you want to plot
    base: The base height of the table (header without any rows)
    height_per_row: The height that one row requires
    char_limit: If the length of a value crosses this limit, the row's height needs to be expanded to fit the value
    height_padding: Extra height in a row when a length of value exceeds char_limit
    '''
    total_height = 0 + base
    for x in range(df.shape[0]):
        total_height += height_per_row
    for y in range(df.shape[1]):
        if len(str(df.iloc[x, y])) > char_limit:
            total_height += height_padding
    return total_height

################### Inputs ###################
# loan_principal          # 10M$ - 500M$ by 250K$ steps
# interest_rate           # 10-12%
# origination_fee         # 0.8-1.0%
#
# start_date              # calendar picker
# duration                # 12*period
# day_count_convention    # 30_360, 30_365, actual_365, actual_365, actual_actual
#
# period                  # 30days => can be derived from the above 3 fields
# days_per_year           # 360/365 => ""     ""              ""
#

################### Input Ranges ###################
pre_set_sel = ['Custom', 'Lava', 'Unchained']
step = 5_000_000
loan_principal_sel = np.arange(10_000_000, 100_000_000 + step, step)
step = 0.005
loan_interest_rate_sel = np.arange(0.10, 0.14 + step, step)
market_interest_rate_sel = np.arange(0.01, 0.14 + step, step)
step = 0.0005
origination_fee_sel = np.arange(0.0080, 0.0200 + step, step)
origination_fee_separate_sel = [False, True]

step = 2_500
btc_price_sel = np.arange(30_000, 300_000 + step, step)

start_date_sel = [pd.Timestamp('2024-01-01'), pd.Timestamp('2025-01-01')]
duration_sel = [12]
day_count_convention_sel = ['30/360', '30/365', 'actual/360', 'actual/365', 'actual/actual']
# day_count_convention_sel = ['30/360', '30/365', 'actual/actual']

# 30/360 => first due date is start + 30 @ 30*IR/360, and so on for duration times
# 30/365 => first due date is start + 30 @ 30*IR/365, and so on for duration times
# actual/360 => first due date is end_of_month(start) @ DaysBetween(start, end_of_month)*IR/360, and so on for duration times
# actual/365 => first due date is end_of_month(start) @ DaysBetween(start, end_of_month)*IR/365, and so on for duration times
# actual/actual => first due date is end_of_month(start) @ DaysBetween(start, end_of_month)*IR/DaysInYear, and so on for duration times

step = 0.005
liquidation_apr_sel = np.arange(0.05, 0.25 + step, step)
mc_to_liq_time_period_sel = ['1-day', '3-days', '1-week']
mc_to_liq_confidence_level_sel = ['95%', '99%', '99.9%', '99.99%']
mc_to_liq_time_horizon_sel = [30, 60, 90, 120]
mc_to_liq_var_not_cvar_sel = [True, False]

init_to_mc_time_period_sel = ['1-week', '2-weeks', '3-weeks', '1-month']
init_to_mc_confidence_level_sel = ['95%', '99%', '99.9%', '99.99%']
init_to_mc_time_horizon_sel = [30, 60, 90, 120]
init_to_mc_var_not_cvar_sel = [True, False]

################### Initial Inputs ###################
pre_set = pre_set_sel[0]
loan_principal = loan_principal_sel[0]

initial_cvl = 1.6
margin_call_cvl = 1.35
liquidation_cvl = 1.25

loan_interest_rate = loan_interest_rate_sel[0]
market_interest_rate = market_interest_rate_sel[9]
origination_fee = origination_fee_sel[0]
origination_fee_separate = origination_fee_separate_sel[0]

btc_price = btc_price_sel[0]

start_date = start_date_sel[0]
duration = duration_sel[0]
day_count_convention = day_count_convention_sel[2]

liquidation_apr = liquidation_apr_sel[round(len(liquidation_apr_sel)/4)]
mc_to_liq_time_period = mc_to_liq_time_period_sel[1]
mc_to_liq_confidence_level = mc_to_liq_confidence_level_sel[0]
mc_to_liq_time_horizon = mc_to_liq_time_horizon_sel[2]
mc_to_liq_var_not_cvar = mc_to_liq_var_not_cvar_sel[-1]
init_to_mc_time_period = init_to_mc_time_period_sel[-1]
init_to_mc_confidence_level = init_to_mc_confidence_level_sel[0]
init_to_mc_time_horizon = init_to_mc_time_horizon_sel[2]
init_to_mc_var_not_cvar = init_to_mc_var_not_cvar_sel[-1]

################### Dashboard ###################
app = Dash(__name__, external_stylesheets=[dbc.themes.FLATLY])
app.title = "Loan Calculator"
fig = go.Figure()

app.layout = dbc.Container([
    html.Div([
        html.Div([
            html.H1([
                html.Span("Simple Interest"),
                html.Br(),
                html.Span("Loan Calculator")
            ]),
            html.P("Prototype showing valuation based on simple-interest loan calculations."),
            html.P("What-if scenarios based on the following selection of inputs.")
            ], style={"vertical-alignment": "top", "height": 230}),
        html.Div([
            html.Div([
                html.H2('Pre Set:'),
                dcc.Dropdown(
                    id="pre-set",
                    options=[{"label": x, "value": x} for x in pre_set_sel],
                    value=pre_set,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Loan Principal:'),
                dcc.Dropdown(
                    id="loan-principal",
                    options=[{"label": locale.currency(x, grouping=True), "value": x} for x in loan_principal_sel],
                    value=loan_principal,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Assumed Disbursement:'),
                dcc.Dropdown(
                    id="loan-disbursement",
                    options=[{"label": locale.currency(x, grouping=True), "value": x} for x in loan_principal_sel],
                    value=loan_principal,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Initial CVL:'),
                dcc.Input(
                    id="initial-cvl",
                    type="number",
                    value=initial_cvl,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Margin Call CVL:'),
                dcc.Input(
                    id="margin-call-cvl",
                    type="number",
                    value=margin_call_cvl,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Liquidation CVL:'),
                dcc.Input(
                    id="liquidation-cvl",
                    type="number",
                    value=liquidation_cvl,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Loan Interest Rate:'),
                dcc.Dropdown(
                    id="loan-interest-rate",
                    options=[{"label": '{:,.2f}%'.format(x * 100), "value": x} for x in loan_interest_rate_sel],
                    value=loan_interest_rate,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Market Interest Rate:'),
                dcc.Dropdown(
                    id="market-interest-rate",
                    options=[{"label": '{:,.2f}%'.format(x * 100), "value": x} for x in market_interest_rate_sel],
                    value=market_interest_rate,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Origination fee:'),
                dcc.Dropdown(
                    id="origination-fee",
                    options=[{"label": '{:,.2f}%'.format(x * 100), "value": x} for x in origination_fee_sel],
                    value=origination_fee,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Is Origination fee separate?:'),
                dcc.RadioItems(
                    id="origination-fee-separate",
                    options=[{"label": f"{x}", "value": x} for x in origination_fee_separate_sel],
                    value=origination_fee_separate,
                ),
            ]),
            html.Div([
                html.H2('BTC Price:'),
                dcc.Dropdown(
                    id="btc-price",
                    options=[{"label": x, "value": x} for x in btc_price_sel],
                    value=btc_price,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Start Date:'),
                dcc.Dropdown(
                    id="start-date",
                    options=[{"label": x, "value": f"{x}"} for x in start_date_sel],
                    value=f"{start_date}",
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Duration:'),
                dcc.Dropdown(
                    id="duration",
                    options=[{"label": x, "value": x} for x in duration_sel],
                    value=duration,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Day Count Convention:'),
                dcc.Dropdown(
                    id="day-count-convention",
                    options=[{"label": x, "value": x} for x in day_count_convention_sel],
                    value=day_count_convention,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),

            html.Br(),

            html.Div([
                html.H2('Log Scale:'),
                dcc.RadioItems(
                    id="log-scale",
                    options=[{"label": f"{x}", "value": x} for x in [True, False]],
                    value=True,
                ),
            ]),

            html.Br(),

            html.Div([
                html.H2('Liquidation Recovery APR:'),
                dcc.Dropdown(
                    id="liquidation-apr",
                    options=[{"label": '{:,.2f}%'.format(x * 100), "value": x} for x in liquidation_apr_sel],
                    value=liquidation_apr,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Margin Call to Liquidation Time Period:'),
                dcc.Dropdown(
                    id="mc-to-liq-time-period",
                    options=[{"label": x, "value": x} for x in mc_to_liq_time_period_sel],
                    value=mc_to_liq_time_period,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Margin Call to Liquidation Confidence Level:'),
                dcc.Dropdown(
                    id="mc-to-liq-confidence-level",
                    options=[{"label": x, "value": x} for x in mc_to_liq_confidence_level_sel],
                    value=mc_to_liq_confidence_level,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Margin Call to Liquidation Time Horizon:'),
                dcc.Dropdown(
                    id="mc-to-liq-time-horizon",
                    options=[{"label": x, "value": x} for x in mc_to_liq_time_horizon_sel],
                    value=mc_to_liq_time_horizon,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Margin Call to Liquidation use VaR? CVaR?:'),
                dcc.RadioItems(
                    id="mc-to-liq-var-not-cvar",
                    options=[{"label": f"{x}", "value": x} for x in mc_to_liq_var_not_cvar_sel],
                    value=True,
                ),
            ]),

            html.Br(),

            html.Div([
                html.H2('Initial to first Margin Call Time Period:'),
                dcc.Dropdown(
                    id="init-to-mc-time-period",
                    options=[{"label": x, "value": x} for x in init_to_mc_time_period_sel],
                    value=init_to_mc_time_period,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Initial to first Margin Call Confidence Level:'),
                dcc.Dropdown(
                    id="init-to-mc-confidence-level",
                    options=[{"label": x, "value": x} for x in init_to_mc_confidence_level_sel],
                    value=init_to_mc_confidence_level,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Initial to first Margin Call Time Horizon:'),
                dcc.Dropdown(
                    id="init-to-mc-time-horizon",
                    options=[{"label": x, "value": x} for x in init_to_mc_time_horizon_sel],
                    value=init_to_mc_time_horizon,
                    clearable=False,
                    optionHeight=40,
                    className='customDropdown'
                ),
            ]),
            html.Div([
                html.H2('Initial to first Margin Call use VaR? CVaR?:'),
                dcc.RadioItems(
                    id="init-to-mc-var-not-cvar",
                    options=[{"label": f"{x}", "value": x} for x in init_to_mc_var_not_cvar_sel],
                    value=True,
                ),
            ])
        ], style={'margin-left': 15, 'margin-right': 15, 'margin-top': 30})
    ], style={
        'width': 340,
        'margin-left': 35,
        'margin-top': 35,
        'margin-bottom': 35
    }),
    html.Div(
        [
            html.Div(
                dcc.Graph(
                    id="graph1",
                    figure=fig
                ),
                style={'width': 950}
            ),
        ],
        style={
            'width': 950,
            'margin-top': 75,
            'margin-right': 35,
            'margin-bottom': 35,
            'display': 'flex'
        })
    ],
    fluid=True,
    style={'display': 'flex'},
    className='dashboard-container'
)

@app.callback(
    Output("graph1", "figure"),
    Input("pre-set", "value"),
    Input("loan-principal", "value"),
    Input("loan-interest-rate", "value"),
    Input("market-interest-rate", "value"),
    Input("origination-fee", "value"),
    Input("origination-fee-separate", "value"),
    Input("initial-cvl", "value"),
    Input("margin-call-cvl", "value"),
    Input("liquidation-cvl", "value"),
    Input("btc-price", "value"),
    Input("start-date", "value"),
    Input("duration", "value"),
    Input("day-count-convention", "value"),
    Input("log-scale", "value"),

    Input("liquidation-apr", "value"),

    Input("mc-to-liq-time-period", "value"),
    Input("mc-to-liq-confidence-level", "value"),
    Input("mc-to-liq-time-horizon", "value"),
    Input("mc-to-liq-var-not-cvar", "value"),

    Input("init-to-mc-time-period", "value"),
    Input("init-to-mc-confidence-level", "value"),
    Input("init-to-mc-time-horizon", "value"),
    Input("init-to-mc-var-not-cvar", "value"),
)
def update_graph1(
        pre_set,
        loan_principal,
        loan_interest_rate,
        market_interest_rate,
        origination_fee,
        origination_fee_separate,
        initial_cvl,
        margin_call_cvl,
        liquidation_cvl,
        btc_price,
        start_date,
        duration,
        day_count_convention,
        log_scale,
        liquidation_apr,
        mc_to_liq_time_period,
        mc_to_liq_confidence_level,
        mc_to_liq_time_horizon,
        mc_to_liq_var_not_cvar,
        init_to_mc_time_period,
        init_to_mc_confidence_level,
        init_to_mc_time_horizon,
        init_to_mc_var_not_cvar,
    ):
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[.12, .83],
        specs=[[{"type": 'xy'}], [{"type": "table"}]],
        horizontal_spacing=0,
        vertical_spacing=0.06
    )

    if pre_set == 'Lava':
        loan_principal = 10_000_000
        loan_interest_rate = 10/100
        market_interest_rate = 5.5/100 # https://tradingeconomics.com/el-salvador/interest-rate
        origination_fee = 0.80/100
        origination_fee_separate = False
        initial_cvl = 160/100
        margin_call_cvl = 135/100
        liquidation_cvl = 125/100
        duration = 12
        day_count_convention = 'actual/360'
    elif pre_set == 'Unchained':
        loan_principal = 150_000
        loan_interest_rate = 14/100
        market_interest_rate = 4.75/100 # https://tradingeconomics.com/united-states/interest-rate
        origination_fee = 2/100
        origination_fee_separate = True
        initial_cvl = 1/(40/100)
        margin_call_cvl = 1/(66.66/100)
        liquidation_cvl = 1/(80/100)
        btc_price = 98167.44
        duration = 12
        day_count_convention = '30/365'

    loan_calc = SimpleLoanCalculator(
        loan_principal,
        loan_interest_rate,
        market_interest_rate,
        origination_fee,
        origination_fee_separate,
        initial_cvl,
        margin_call_cvl,
        liquidation_cvl,
        btc_price,
        pd.Timestamp(start_date),
        duration,
        str(day_count_convention)
    )

    loan_calc_df = loan_calc.table_df()

    cvl_calc = CvlCalculator(
        # loan_calc.apr_no_origination_fee,
        liquidation_apr,
        mc_to_liq_time_period,
        mc_to_liq_confidence_level,
        mc_to_liq_time_horizon,
        mc_to_liq_var_not_cvar,
        init_to_mc_time_period,
        init_to_mc_confidence_level,
        init_to_mc_time_horizon,
        init_to_mc_var_not_cvar
    )

    cvl_calc_df = cvl_calc.table_df()

    table_df = pd.concat([loan_calc_df, cvl_calc_df], ignore_index=True, sort=False)

    # print(loan_calc_df)
    # print(cvl_calc_df)
    # print(table_df)

    fig.add_trace(go.Table(
                    header=dict(values = list(table_df.columns), align="left"),
                    cells=dict(values=[table_df.Field, table_df.Value], align="left"),
                    # header_values=table_df.columns,
                    # cells_values=[table_df[k].tolist() for k in table_df.columns],
                    ),
                    row=2, col=1)

    schedule_df = loan_calc.schedule_df()
    fig.add_trace(go.Bar(x=schedule_df['PeriodDueDate'],
                    y=schedule_df['InterestDue'],
                    text=schedule_df['InterestDue'],
                    # shared_yaxis=False,
                    # xaxis='x2', yaxis='y2',
                    marker=dict(color='#FF3300'),
                    name='Interest Due',
                    ),
                    row=1, col=1)
    fig.add_trace(go.Bar(x=schedule_df['PeriodDueDate'],
                    y=schedule_df['Disbursement'],
                    text=schedule_df['Disbursement'],
                    # xaxis='x2', yaxis='y2',
                    marker=dict(color='#404040'),
                    name='Disbursement',
                    ),
                    row=1, col=1)
    fig.add_trace(go.Bar(x=schedule_df['PeriodDueDate'],
                    y=schedule_df['Payment'],
                    text=schedule_df['Payment'],
                    # xaxis='x2', yaxis='y2',
                    marker=dict(color='#99ff00'),
                    name='Payment',
                    ),
                    row=1, col=1)

    if log_scale:
        fig.update_yaxes(type="log")

    # Update the margins to add a title and see graph x-labels.
    fig.layout.margin.update({'t':75, 'l':50})

    fig.update_traces(
        # cells_font_family="Poppins",
        cells_font_size=14,
        cells_height=22,
        # header_font_family="Poppins",
        header_font_size=16,
        selector=dict(type='table'))

    fig.update_layout(
        # paper_bgcolor='#6d6d71',
        # plot_bgcolor='#6d6d71',
        # plot_bgcolor='#ffffff',
        width=950,
        height=1940,
        # height=calc_table_height(table_df, height_per_row=23),
        showlegend=True,
        # margin=dict(l=0,r=0,t=0,b=0)
    )
    return fig

# if __name__ == "__main__":
#    # Turn off reloader if inside Jupyter
#    app.run(debug=True, port=8051, use_reloader=True)

application = app.server
