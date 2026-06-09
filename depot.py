import csv
import requests
import yfinance as yf
import math #test, isNan
from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np # depot total as nansum()
import json

currency = "€"
toEUR = 1/1.1669 #side do: call exchange
source = "yahoo" #source todo: yahoo or tradegate

@dataclass
class Stock:
    name: str
    symbol: str
    amount: float
    initial: float # average price (incl. fees) from csv file
    current: float # online price
    payout: float
    currency = "€"
    changeRel= 0.0
    changeAbs: float
    sRel = "normal"
    age = 0.05 #in days
    isSold = False
    isSingle = True
    len = 0
    orders = []
    def fetch(self, period="1mo"):
        if (self.symbol == "None") or (self.symbol == "const"):
            self.changeAbs  = self.amount * (self.payout)
            self.changeRel  = self.changeAbs / self.initial
            self.current    = self.initial
            self.len = 0
            return None
        try: #todo:handle  Yahoo error = "No data found, symbol may be delisted
            stock = yf.Ticker(self.symbol)
            data = stock.history(period=period)
            self.data = data            # todo: if today is not in history file.
            self.current =  data['Close'].iloc[-1]# side do: update interval
            self.len = len(data)
        except Exception as e:
            self.current = self.initial # todo: last known value.
            print(f"Error fetching data: {str(e)}")
        delta = self.payout + self.current - self.initial
        self.changeAbs  = self.amount * (delta)
        self.changeRel  = delta / self.initial #not used for const bonds
        return data

    def average(self, days): #side do: start = 0
        a = 0
        avg=0
        try:
            self.len = len(self.data)
        except Exception as e:
            print(f"Fetching data for averaging")
            return self.initial #for bonds no self.data exists, symbol=="const"
        if self.len < days: #is data insufficient?
            if days > 440:
                self.fetch("5y")
            elif days > 220:
                self.fetch("2y")
            elif days > 95:
                self.fetch("1y")
            elif days > 21:   #workingdays e.g. February
                self.fetch("6mo")
            elif days >5 :
                self.fetch("1mo")
            else:
                self.fetch("5d")
        nr=0
        for d in range(-days-1, -1): # exclude today [-1]
            nr +=1
            try:
                a += self.data['Close'].iloc[d] #todo, if stock held
            except Exception as e:
                a+= self.initial    # assume no change, if wrong s.symbol
                print(f"Error averaging {self.name} data of {days+2-nr} days ago: {str(e)}, max {len(self.data)} values")
        avg = a / nr
        return avg

@dataclass
class Order:
    name: str
    symbol: str
    amount: float
    action: str  # "buy", "sell" or "payout" for dividends and coupons
    price: float
    date: str # for highlighting stock holding time range
    datesell: str #Optional when sold (completely)

class Consors: #remote depot
    def consors(self):
        url = "https://apiconsorsbank.de/trading/v1/ex-ante-costs/id"
        headers = {"accept": "application/hal+json;charset=UTF-8"}
        response = rq.get(url, headers=headers)
        print(response.text)
    def posHistory(self):
        url = "https://api.consorsbank.de/sandbox/trading/v1/securities-accounts/no/positions-histories"
        headers = {
            "accept": "application/hal+json",
            "authorization": "Bearer test"
        }
        response = requests.get(url, headers=headers)
        print(response.text)

class Depot: #local copy in csv file
    def __init__(self, depot_csv: str, order_csv: str, ticker_mapping: dict):
        self.depot_path = depot_csv
        self.order_path = order_csv
        self.stocks: list[Stock] = []
        self.orders: list[Order] = []
        self.read_order()
        self.name = ""
        self.total= 0.0
        self.initial = 0.0
        self.changeRel = 0
        self.changeAbs = 0
        self.currency = "€"
        self.read()
        #todo: catch empty s.current
        for s in self.stocks:
            i             = s.initial * s.amount
            self.initial += i # total sum, no dividend payout
            s.current     = s.initial
            try:
                val       = (s.payout+s.current) * s.amount
            except Exception:
                val       = (s.payout+s.initial) * s.amount #wrong symbol
            self.total   += val
            self.changeAbs += val - s.initial
        if self.initial == 0:
            self.changeRel = 0
        else:
            self.changeRel = self.total/self.initial
        return None
    
    def intraday(self, symbol: str):
        #self.read() # from csv file
        total = 0.0
        self.changeAbs = 0
        self.changeRel = 0
        self.payout    = 0
        t = "Test from depot.update(): "
        #initial = 1000 # todo read from file
        for s in self.stocks: # search matching symbol
            #add if s.symbol == symbol:
            s.fetch()
            delta = s.payout + s.current - s.initial
            self.changeAbs += s.amount * (s.changeAbs + s.payout)
            self.changeRel = delta / s.initial #e.g 1.08 for 8%
            self.payout += s.payout
            val    = s.amount *(s.current+s.payout) # may be NaN
            old    = s.amount *(s.initial+s.payout)
            total += val if isinstance(val, float) else old # newer value
            print(f"Dividend: {self.payout}€")
        self.changeRel = total / self.initial
        t += f"{total:.0f}"
        self.total = total
        self.changeAbs = self.total - self.initial  
        self.changeRel = self.changeAbs / self.initial
        return t
                
    def update(self):
        #self.read() # from csv file
        total = 0.0
        self.changeAbs = 0
        self.changeRel = 0
        self.payout    = 0
        nr=0
        t = "Test from depot.update(): "
        #initial = 1000 # todo read from file
        for s in self.stocks: # from csv file
            try: #skips stock symbol/alias not found
                s.fetch()
                s.current = s.data['Close'].iloc[-1] #last closing price
            except Exception as e:   #if s.current does not exist
                s.current = s.initial
                if not(s.symbol == "const" or s.symbol == "None"):
                   print(f"Error fetching data for {s.name}. Symbol {s.symbol} may be wrong: {str(e)}")
            delta = s.payout + s.current - s.initial
            self.changeAbs += s.amount * (s.changeAbs + s.payout)
            self.changeRel = delta / s.initial #e.g 1.08 for 8%
            self.payout += s.payout
            val    = s.amount *(s.current+s.payout)
            old    = s.amount *(s.initial+s.payout)
            total += val if isinstance(val, float) else old # newer value
            nr += 1
            #print(f"Dividend nr. {nr}: {s.payout} {currency}")
        t += f"{total:.0f}"
        self.total = total
        self.changeAbs = self.total - self.initial  
        if self.initial == 0:
            self.changeRel = 0
        else:
            self.changeRel = self.changeAbs/self.initial
        return t

    def past_profit(self): # stocks already sold
        profit = 0.0
        nr=0
        pastStocks=[]
        t = "Test from depot.past_profit(): "
        for o in self.orders: # from csv file
            if not (o.alias in pastStocks):
                stocks.append(o.alias) #unique stocks
                #dictionary with + and -
            # check, if more stocks are sold than bought -> sold = bought
            # je sell, passendes buy finden
            # falls differenz negativ meldung, aber limit auf kaufmenge
            # falls differenz amount 0 ist ok. Sonst noch gehalten
        print(t+str(stocks))

        for s in pastStocks: # all stocks (in orders), not only self.stocks
            try:#skips symbol, if not sold
                s.current = s.data['Close'].iloc[-1] #last closing price
                delta = s.payout + s.current - s.initial
                self.changeAbs += s.amount * (s.changeAbs + s.payout)
                self.changeRel = delta / s.initial #e.g 1.08 for 8%
                self.payout += s.payout
                val    = s.amount *(s.current+s.payout) # may be NaN
                old    = s.amount *(s.initial+s.payout)
                total += val if isinstance(val, float) else old # newer value
                nr += 1
                #print(f"Dividend nr. {nr}: {s.payout} {currency}")
            except Exception as e:
                print(f"Error fetching data for {s.name}: {str(e)}")
        t += f"{total:.0f}"
        self.total = total
        self.changeAbs = self.total - self.initial  
        if self.initial == 0:
            self.changeRel = 0
        else:
            self.changeRel = self.changeAbs/self.initial
        return t

    def data(self, symbol: str, data: dict):
        t="Data was not needed in depot."
        for s in self.stocks:
            if s.symbol == symbol:
                s.data = data # stock data is handed, setter is resolved.
                t = f"Data for {s.name} is handed over to class."
        return t

    def read_depot(self):
        """Lädt die Depotdaten aus der tabulatorgetrennten CSV-Datei."""
        with open(self.depot_path, mode='r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                #increaseOpt = float(row['changeRel']) if 'changeRel' in row and row['changeRel'] else None
                stock = Stock(
                    name=row['name'],
                    symbol=row['symbol'],
                    amount=float(row['amount']),
                    initial=float(row['initial']),
                    current=float(row['initial']), #todo: same value
                    payout=float(row['payout']),
                    changeAbs = 0.0
                )
                self.stocks.append(stock)
        return None

    def read_order(self):
        """Lädt die Orderdaten aus der tabulatorgetrennten CSV-Datei."""
        with open(self.order_path, mode='r', encoding='utf-8', newline='') as file:
            reader = csv.DictReader(file, delimiter='\t')
            for row in reader:
                order = Order(
                    name=row['name'],
                    symbol=row['symbol'],
                    amount=float(row['amount']),
                    price=float(row['price']),
                    action=row['action'],
                    date=row['date'],
                    datesell=row['datesell']
                    )
                self.orders.append(order)
        return None

    def read(self):
        t = "OK. Read depot and order files."
        try:
            self.read_depot()
        except Exception as e:
            t = f"Error reading depot csv files: {str(e)}"
            print(t)
        try:
            self.read_order()
        except Exception as e:
            t = f"Error reading order csv files: {str(e)}"
            print(t)
        return t
 
    def show(self):
        """Zeigt alle stocks im Depot an."""
        text = f"___ Depot: {self.total:.0f} {currency} ___\n"
        for s in self.stocks:
            text+=f"{s.name}: {s.amount:0} shares changed from: {s.initial:.2f} "
            text+=f"to {s.current:.2f}, incl. {s.payout:.2f} dividend "
            text+=f"increase is {s.changeRel*100:.1f}% / {s.changeAbs:.0f}{s.currency}\n"
        print(text)
        return text

    def show_orders(self):
        """Zeigt alle Orders an."""
        for order in self.orders:
            print(f"{s.name} {order.amount} {order.action} for {order.price} at {order.datum}")

    def get_price(self, symbol: str) -> Optional[float]:
        """
        Price from stock markets (yahoo, tradegate) and writes relative changeRel
        """
        for stock in self.stocks:
            if stock.symbol == symbol:
                #source todo: yahoo or tradegate
                current_price = stock. initial * 1.10  # Beispiel: 10% Wertzuwachs
                changeRel = ((current_price - stock. initial) / stock. initial) * 100
                stock.changeRel = changeRel
                self.schreibe_depot_daten()
                #self.calc_total()
                return current_price
        return None

    def schreibe_depot_daten(self):
        """Schreibt die aktualisierten Depotdaten zurück in die CSV-Datei."""
        with open(self.depot_pfad, mode='w', encoding='utf-8', newline='') as file:
            writer = csv.DictWriter(file, delimiter='\t', fieldnames=['name', 'symbol', 'amount', ' initial', 'changeRel'])
            writer.writeheader()
            for stock in self.stocks:
                writer.writerow({
                    'name': stock.name,
                    'symbol': stock.symbol,
                    'amount': stock.amount,
                    ' initial': stock.initial,
                    'changeRel': stock.changeRel if stock.changeRel is not None else ''
                })

    def stock_ranges(self) -> list[dict[str, tuple[str, float]]]:
        """
        Gibt ein Array mit Zeitdauern und prozentualen Gewinnen pro Aktie zurück.
        Beispiel: [{'Aktie A': ('1 Jahr', 10.0), 'Aktie B': ('6 Monate', 5.5)}]
        """
        result = []
        for stock in self.stocks:
            zeitdauer = "1 Jahr"  # Beispielwert, anpassen nach Bedarf
            prozentualer_gewinn = stock.changeRel if stock.changeRel is not None else 0.0
            result.append({stock.name: (zeitdauer, prozentualer_gewinn)})
        return result

#depot = Depot("depot.csv", "order.csv", ticker_mapping)

