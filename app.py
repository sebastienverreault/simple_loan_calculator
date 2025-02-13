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
        print(self.interest_rate)
        self.time_cash_flows_df = time_cash_flows_df
        print(self.time_cash_flows_df)

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
            print(f"#days used for '30/{day_per_year}': {result}")
            result = result / day_per_year
        else:
            raise RuntimeError(f"Unsupported convention: '{convention}'")
        return result

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

    table_df = loan_calc.table_df()
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
        height=1240,
        # height=calc_table_height(table_df, height_per_row=23),
        showlegend=True,
        # margin=dict(l=0,r=0,t=0,b=0)
    )
    return fig

#if __name__ == "__main__":
#    # Turn off reloader if inside Jupyter
#    app.run_server(debug=True, port=8051, use_reloader=True)

application = app.server
